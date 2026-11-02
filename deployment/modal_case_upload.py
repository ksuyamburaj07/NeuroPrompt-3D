"""NeuroPrompt-3D protected cloud MRI ingestion.

Bounded, research-only staging service.
No model execution and no public anonymous uploads.
"""

import modal

app = modal.App("neuroprompt3d-case-upload")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.14.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "fastapi==0.142.2",
        "numpy==2.4.6",
        "nibabel==5.4.2",
        "scipy==1.17.1",
    )
    .add_local_dir("src", "/root/src")
    .add_local_dir(
        "app/backend/core",
        "/root/app/backend/core",
    )
    .add_local_dir(
        "app/backend/schemas",
        "/root/app/backend/schemas",
    )
    .add_local_dir(
        "app/backend/services",
        "/root/app/backend/services",
    )
)

runtime = modal.Volume.from_name("np3d-live-runtime")

CHUNK_BYTES = 768 * 1024
MAX_MODALITY_BYTES = 80 * 1024 * 1024
MAX_CASE_BYTES = 400 * 1024 * 1024

REQUIRED_MODALITIES = ("t1n", "t1c", "t2w", "t2f")

UPLOAD_ROOT = (
    "/root/app/backend/runtime/.pending_uploads"
)


@app.function(
    image=image,
    cpu=1,
    memory=4096,
    max_containers=1,
    volumes={
        "/root/app/backend/runtime": runtime,
    },
)
def verify_upload_environment():
    import importlib.metadata as md
    from pathlib import Path

    import torch

    from app.backend.services.case_service import (
        validate_and_stage_case,
    )

    assert callable(validate_and_stage_case)
    assert torch.__version__.split("+")[0] == "2.14.0"

    return {
        "torch": torch.__version__,
        "fastapi": md.version("fastapi"),
        "nibabel": md.version("nibabel"),
        "chunk_bytes": CHUNK_BYTES,
        "max_modality_bytes": MAX_MODALITY_BYTES,
        "max_case_bytes": MAX_CASE_BYTES,
        "required_modalities": REQUIRED_MODALITIES,
        "runtime_root": str(Path(UPLOAD_ROOT).parent),
        "original_validator_available": True,
    }


@app.local_entrypoint()
def verify():
    result = verify_upload_environment.remote()

    assert result["original_validator_available"]

    print("PASS: Original MRI validator imports in Modal")
    print("PASS: Persistent live-runtime Volume mounted")
    print("PASS: Upload size limits configured")
    print(result)


# Batch 8: authenticated, bounded upload sessions.
# The scientific validator above remains unchanged.

@app.function(
    image=image,
    cpu=1,
    memory=4096,
    timeout=300,
    max_containers=1,
    volumes={
        "/root/app/backend/runtime": runtime,
    },
)
@modal.asgi_app(requires_proxy_auth=True)
def upload_api():
    import asyncio
    import hashlib
    import json
    import re
    import secrets
    import shutil
    import time

    from contextlib import ExitStack
    from pathlib import Path
    from uuid import uuid4

    from fastapi import FastAPI, HTTPException, Request, Query, UploadFile
    from fastapi.responses import JSONResponse

    from app.backend.services.case_service import (
        validate_and_stage_case,
    )

    api = FastAPI(
        title="Protected NeuroPrompt-3D MRI Uploads",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    root = Path(UPLOAD_ROOT)
    session_pattern = re.compile(r"up_[0-9a-f]{32}")
    sha_pattern = re.compile(r"[0-9a-f]{64}")
    case_pattern = re.compile(r"case_[0-9a-f]{32}")
    fields = set(REQUIRED_MODALITIES) | {"segmentation"}

    max_sessions = 2
    max_saved_cases = 8
    lifetime_seconds = 2 * 60 * 60

    mutex = asyncio.Lock()

    def session_directory(session_id):
        if not session_pattern.fullmatch(session_id):
            raise HTTPException(400, "Invalid upload session")
        return root / session_id

    def save_metadata(directory, data):
        temporary = directory / ".session.tmp"
        temporary.write_text(json.dumps(data, sort_keys=True))
        temporary.replace(directory / "session.json")

    def load_session(request, session_id):
        directory = session_directory(session_id)
        metadata = directory / "session.json"

        if not metadata.is_file():
            raise HTTPException(404, "Upload session not found")

        data = json.loads(metadata.read_text())
        token = request.headers.get("x-np3d-session-token", "")
        digest = hashlib.sha256(token.encode()).hexdigest()

        if not secrets.compare_digest(
            digest, data["token_sha256"]
        ):
            raise HTTPException(403, "Invalid upload session token")

        if time.time() >= data["expires_at"]:
            raise HTTPException(410, "Upload session expired")

        return directory, data

    def describe(data):
        return {
            "session_id": data["session_id"],
            "status": data["status"],
            "expires_at": data["expires_at"],
            "offsets": {
                name: spec["offset"]
                for name, spec in data["files"].items()
            },
            "validation": data.get("validation"),
        }

    def cleanup_expired():
        root.mkdir(parents=True, exist_ok=True)

        for directory in root.iterdir():
            if directory.is_symlink() or not directory.is_dir():
                continue

            if not session_pattern.fullmatch(directory.name):
                continue

            metadata = directory / "session.json"

            try:
                data = json.loads(metadata.read_text())
                expired = time.time() >= data["expires_at"]
            except (OSError, ValueError, KeyError):
                continue

            if expired:
                shutil.rmtree(directory)

    @api.post("/api/v1/upload-sessions")
    async def create_session(request: Request):
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid session request") from None

        if not isinstance(payload, dict):
            raise HTTPException(400, "Invalid upload manifest")

        supplied = payload.get("files")

        if not isinstance(supplied, dict):
            raise HTTPException(422, "Missing file manifest")

        if not set(REQUIRED_MODALITIES).issubset(supplied):
            raise HTTPException(422, "Four MRI modalities required")

        if not set(supplied).issubset(fields):
            raise HTTPException(422, "Unknown modality")

        total = 0
        specs = {}

        for name, item in supplied.items():
            if not isinstance(item, dict):
                raise HTTPException(422, "Invalid file metadata")

            filename = item.get("filename")
            size = item.get("size")
            digest = item.get("sha256")

            if (
                not isinstance(filename, str)
                or len(filename) > 160
                or "/" in filename
                or "\\" in filename
                or filename in {".", ".."}
                or not filename.lower().endswith((".nii", ".nii.gz"))
                or type(size) is not int
                or size < 1
                or size > MAX_MODALITY_BYTES
                or not isinstance(digest, str)
                or sha_pattern.fullmatch(digest) is None
            ):
                raise HTTPException(422, f"Invalid {name} metadata")

            total += size

            specs[name] = {
                "filename": filename,
                "size": size,
                "sha256": digest,
                "offset": 0,
            }

        if total > MAX_CASE_BYTES:
            raise HTTPException(413, "MRI case exceeds size limit")

        async with mutex:
            runtime.reload()
            cleanup_expired()

            active = sum(
                1 for path in root.iterdir()
                if path.is_dir() and
                session_pattern.fullmatch(path.name)
            )

            if active >= max_sessions:
                raise HTTPException(
                    429, "Upload session capacity reached"
                )

            cases_root = Path(
                "/root/app/backend/runtime/live_cases"
            )

            existing_cases = (
                sum(
                    1 for p in cases_root.iterdir()
                    if p.is_dir() and
                    case_pattern.fullmatch(p.name)
                )
                if cases_root.is_dir() else 0
            )

            if existing_cases >= max_saved_cases:
                raise HTTPException(
                    429, "Saved case capacity reached"
                )

            session_id = "up_" + uuid4().hex
            token = secrets.token_urlsafe(32)

            directory = root / session_id
            directory.mkdir(parents=True, exist_ok=False)

            data = {
                "session_id": session_id,
                "token_sha256": hashlib.sha256(
                    token.encode()
                ).hexdigest(),
                "expires_at": time.time() + lifetime_seconds,
                "status": "uploading",
                "files": specs,
                "validation": None,
            }

            save_metadata(directory, data)
            runtime.commit()

        return JSONResponse(
            status_code=201,
            content={**describe(data), "token": token},
            headers={"Cache-Control": "private, no-store"},
        )

    @api.get("/api/v1/upload-sessions/{session_id}")
    async def session_status(session_id: str, request: Request):
        async with mutex:
            runtime.reload()
            _, data = load_session(request, session_id)
            return describe(data)

    @api.put(
        "/api/v1/upload-sessions/{session_id}/files/{field}"
    )
    async def upload_chunk(
        session_id: str,
        field: str,
        request: Request,
        offset: int = Query(ge=0),
    ):
        if field not in fields:
            raise HTTPException(404, "Unknown modality")

        chunks = []
        total = 0

        async for block in request.stream():
            total += len(block)
            if total > CHUNK_BYTES:
                raise HTTPException(413, "MRI chunk exceeds limit")
            chunks.append(block)

        payload = b"".join(chunks)

        if not payload:
            raise HTTPException(400, "Empty MRI chunk")

        async with mutex:
            runtime.reload()
            directory, data = load_session(request, session_id)

            if data["status"] != "uploading":
                raise HTTPException(409, "Session not accepting data")

            if field not in data["files"]:
                raise HTTPException(404, "File not declared")

            spec = data["files"][field]
            current = spec["offset"]
            path = directory / (field + ".part")

            if offset < current:
                if offset + len(payload) > current:
                    raise HTTPException(409, "Overlapping chunk")

                with path.open("rb") as existing:
                    existing.seek(offset)
                    identical = existing.read(len(payload)) == payload

                if not identical:
                    raise HTTPException(409, "Chunk replay mismatch")

                return {"offset": current, "replayed": True}

            if offset != current:
                raise HTTPException(409, "Unexpected upload offset")

            if offset + len(payload) > spec["size"]:
                raise HTTPException(413, "File exceeds declared size")

            with path.open("ab") as output:
                output.write(payload)
                output.flush()

                import os
                os.fsync(output.fileno())

            spec["offset"] += len(payload)

            if spec["offset"] == spec["size"]:
                digest = hashlib.sha256()

                with path.open("rb") as source:
                    for block in iter(
                        lambda: source.read(1024 * 1024), b""
                    ):
                        digest.update(block)

                if digest.hexdigest() != spec["sha256"]:
                    data["status"] = "integrity_failed"
                    save_metadata(directory, data)
                    runtime.commit()
                    raise HTTPException(
                        422, "Uploaded MRI checksum mismatch"
                    )

            save_metadata(directory, data)
            runtime.commit()

            return {
                "offset": spec["offset"],
                "complete": spec["offset"] == spec["size"],
            }

    @api.post(
        "/api/v1/upload-sessions/{session_id}/finalize"
    )
    async def finalize_session(session_id: str, request: Request):
        async with mutex:
            runtime.reload()
            directory, data = load_session(request, session_id)

            if data["status"] == "finalized":
                return data["validation"]

            if data["status"] != "uploading":
                raise HTTPException(409, "Upload session invalid")

            if any(
                item["offset"] != item["size"]
                for item in data["files"].values()
            ):
                raise HTTPException(
                    409, "MRI uploads are incomplete"
                )

            with ExitStack() as stack:
                uploads = {}

                for name in REQUIRED_MODALITIES:
                    spec = data["files"][name]
                    opened = stack.enter_context(
                        (directory / (name + ".part")).open("rb")
                    )
                    uploads[name] = UploadFile(
                        file=opened,
                        filename=spec["filename"],
                    )

                segmentation = None

                if "segmentation" in data["files"]:
                    spec = data["files"]["segmentation"]
                    opened = stack.enter_context(
                        (directory / "segmentation.part").open("rb")
                    )
                    segmentation = UploadFile(
                        file=opened,
                        filename=spec["filename"],
                    )

                result = await validate_and_stage_case(
                    uploads=uploads,
                    segmentation=segmentation,
                )

            data["validation"] = result.model_dump(mode="json")
            data["status"] = "finalized"

            # The original validator has already moved valid cases
            # into live_cases. Delete only temporary upload copies.
            for name in data["files"]:
                (directory / (name + ".part")).unlink(
                    missing_ok=True
                )

            save_metadata(directory, data)
            runtime.commit()

            return data["validation"]


    @api.delete("/api/v1/cases/{case_id}")
    async def discard_case(case_id: str, request: Request):
        if not case_pattern.fullmatch(case_id):
            raise HTTPException(400, "Invalid case identifier")

        upload_id = request.headers.get("x-np3d-upload-id", "")

        async with mutex:
            runtime.reload()
            directory, data = load_session(request, upload_id)

            validation = data.get("validation") or {}

            if validation.get("case_id") != case_id:
                raise HTTPException(403, "Case/session mismatch")

            if data["status"] == "discarded":
                return {"deleted": True}

            if data["status"] != "finalized":
                raise HTTPException(409, "Case not finalized")

            from app.backend.services.case_service import (
                delete_live_case,
            )
            from app.backend.services.run_service import (
                load_live_run,
            )

            runs_root = Path(
                "/root/app/backend/runtime/live_runs"
            )

            if runs_root.is_dir():
                for entry in runs_root.iterdir():
                    if not (
                        entry.is_dir()
                        and re.fullmatch(
                            r"run_[0-9a-f]{32}", entry.name
                        )
                    ):
                        continue

                    try:
                        record = load_live_run(entry.name)
                    except (OSError, ValueError):
                        raise HTTPException(
                            409, "Run integrity cannot be confirmed"
                        ) from None

                    if record.case_id == case_id:
                        raise HTTPException(
                            409, "Saved run references this case"
                        )

            try:
                delete_live_case(case_id)
            except FileNotFoundError:
                raise HTTPException(404, "Case unavailable") from None

            data["status"] = "discarded"
            save_metadata(directory, data)
            runtime.commit()

            return {"deleted": True}

    return api
