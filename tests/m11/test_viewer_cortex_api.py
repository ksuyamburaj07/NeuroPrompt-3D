"""Case identity, integrity and QC gates for external cortical assets."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backend.core import paths
from app.backend.main import app


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _nifti_bytes(path: Path, value: float) -> bytes:
    data = np.zeros((6, 5, 4), dtype=np.float32)
    data[1:5, 1:4, 1:3] = value
    nib.save(nib.Nifti1Image(data, np.diag([1.25, 1.5, 2.0, 1.0])), path)
    return path.read_bytes()


def _stage(client: TestClient, payload: bytes) -> str:
    response = client.post(
        "/api/v1/cases/validate",
        files={
            field: (f"{field}.nii.gz", payload, "application/gzip")
            for field in ("t1n", "t1c", "t2w", "t2f")
        },
    )
    assert response.status_code == 200
    assert response.json()["valid"] is True
    return response.json()["case_id"]


def _entry(root: Path, reconstruction_id: str, source: bytes) -> dict:
    root.mkdir()
    meshes = {}
    for hemisphere, offset in (("lh", -1), ("rh", 1)):
        # Small PLY fixtures verify transport/provenance, not extraction.
        mesh = (
            "ply\nformat ascii 1.0\n"
            f"comment {reconstruction_id} physical RAS+ {hemisphere}\n"
            "element vertex 3\nproperty float x\nproperty float y\n"
            "property float z\nelement face 1\n"
            "property list uchar int vertex_indices\nend_header\n"
            f"{offset} 0 0\n{offset} 1 0\n{offset} 0 1\n3 0 1 2\n"
        ).encode("ascii")
        filename = f"{hemisphere}.pial.ras.ply"
        (root / filename).write_bytes(mesh)
        meshes[hemisphere] = {"path": filename, "sha256": _digest(mesh)}
    return {
        "reconstruction_id": reconstruction_id,
        "source_t1n_sha256": _digest(source),
        "asset_root": str(root),
        "coordinate_convention": "physical RAS+",
        "units": "mm",
        "qc": {
            "status": "approved",
            "scope": "research",
            "clinically_validated": False,
            "notes": "Approved for synthetic transport tests only.",
        },
        "meshes": meshes,
    }


def _write_registry(context) -> None:
    context.registry_path.write_text(json.dumps(context.registry), encoding="utf-8")


def _status(context, case_id: str | None = None) -> dict:
    case_id = case_id or context.case_a
    response = context.client.get(f"/api/v1/cases/{case_id}/viewer/cortex/status")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    return response.json()


def _mesh(context, hemisphere="lh", case_id: str | None = None):
    case_id = case_id or context.case_a
    return context.client.get(f"/api/v1/cases/{case_id}/viewer/cortex/{hemisphere}")


def _assert_unavailable(context, reason: str) -> dict:
    status = _status(context)
    assert status["available"] is False
    assert status["reason"] == reason
    assert _mesh(context, "lh").status_code == 404
    assert _mesh(context, "rh").status_code == 404
    return status


@pytest.fixture
def cortex_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    live_cases = tmp_path / "live_cases"
    registry_path = tmp_path / "cortex-registry.json"
    monkeypatch.setattr(paths, "LIVE_CASES_ROOT", live_cases)
    monkeypatch.setattr(paths, "LIVE_RUNS_ROOT", tmp_path / "live_runs")
    monkeypatch.setattr(paths, "CORTEX_REGISTRY_PATH", registry_path)
    source_a = _nifti_bytes(tmp_path / "source-a.nii.gz", 100)
    source_b = _nifti_bytes(tmp_path / "source-b.nii.gz", 200)
    entry_a = _entry(tmp_path / "assets-a", "trial-a", source_a)
    entry_b = _entry(tmp_path / "assets-b", "trial-b", source_b)
    with TestClient(app) as client:
        context = SimpleNamespace(
            client=client,
            live_cases=live_cases,
            registry_path=registry_path,
            registry={"schema_version": 1, "enabled": True, "entries": [entry_a, entry_b]},
            source_a=source_a,
            source_b=source_b,
            entry_a=entry_a,
            entry_b=entry_b,
            case_a=_stage(client, source_a),
            case_b=_stage(client, source_b),
        )
        _write_registry(context)
        yield context


def test_multiple_reconstructions_follow_source_identity(cortex_cases):
    context = cortex_cases
    for case_id, entry in ((context.case_a, context.entry_a), (context.case_b, context.entry_b)):
        status = _status(context, case_id)
        assert status["case_id"] == case_id
        assert status["available"] is True
        assert status["reason"] == "available"
        assert status["reconstruction_id"] == entry["reconstruction_id"]
        assert status["coordinate_convention"] == "physical RAS+"
        assert status["units"] == "mm"
        assert status["qc"] == entry["qc"]
        assert "asset_root" not in status
        assert str(Path(entry["asset_root"]).parent) not in json.dumps(status)
        for hemisphere in ("lh", "rh"):
            response = _mesh(context, hemisphere, case_id)
            assert response.status_code == 200
            expected = Path(entry["asset_root"]) / entry["meshes"][hemisphere]["path"]
            assert response.content == expected.read_bytes()
            assert response.headers["content-type"] == "application/octet-stream"
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-mesh-coordinates"] == "RAS+ millimetres"
    # Application UUIDs are staging identifiers, not a cortex allowlist.
    duplicate_case = _stage(context.client, context.source_a)
    assert duplicate_case != context.case_a
    assert _status(context, duplicate_case)["reconstruction_id"] == "trial-a"
    assert _mesh(context, case_id=duplicate_case).content == _mesh(context).content


def test_equal_geometry_and_filenames_never_match_a_different_source(cortex_cases):
    context = cortex_cases
    context.registry["entries"] = [context.entry_a]
    _write_registry(context)
    status = _status(context, context.case_b)
    assert status["available"] is False
    assert status["reason"] == "source_unregistered"
    assert status["reconstruction_id"] is None
    assert status["qc"] is None
    assert _mesh(context, case_id=context.case_b).status_code == 404
    assert _mesh(context).status_code == 200


def test_staged_source_mutation_fails_closed(cortex_cases):
    context = cortex_cases

    # The original validated case is initially eligible.
    assert _mesh(context).status_code == 200

    source_path = (
        context.live_cases
        / context.case_a
        / "modalities"
        / "t1n.nii.gz"
    )

    # Replace A's staged MRI with B's bytes.
    # A's validation-time fingerprint must not change.
    source_path.write_bytes(context.source_b)

    status = _assert_unavailable(
        context,
        "source_identity_mismatch",
    )

    assert status["reconstruction_id"] is None

    # The genuine B case must remain unaffected.
    assert _status(
        context, context.case_b
    )["available"] is True

    assert _mesh(
        context, case_id=context.case_b
    ).status_code == 200

def test_legacy_case_without_source_fingerprint(cortex_cases):
    context = cortex_cases

    metadata_path = (
        context.live_cases
        / context.case_a
        / "case.json"
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    # Simulate a case staged before B7C existed.
    metadata.pop("source_t1n_sha256", None)

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    _assert_unavailable(context, "source_unavailable")

    # Legacy MRI visualization must still work.
    response = context.client.get(
        f"/api/v1/cases/{context.case_a}/viewer/metadata"
    )

    assert response.status_code == 200

@pytest.mark.parametrize("hemisphere", ["lh", "rh"])
def test_corrupt_mesh_disables_both_hemispheres_after_success(cortex_cases, hemisphere):
    context = cortex_cases
    assert _mesh(context).status_code == 200
    mesh_path = Path(context.entry_a["asset_root"]) / f"{hemisphere}.pial.ras.ply"
    original = mesh_path.read_bytes()
    mesh_path.write_bytes(original.replace(b"3 0 1 2", b"3 0 2 1"))
    _assert_unavailable(context, "asset_integrity_failed")


@pytest.mark.parametrize("hemisphere", ["lh", "rh"])
def test_missing_mesh_disables_both_hemispheres(cortex_cases, hemisphere):
    context = cortex_cases
    (Path(context.entry_a["asset_root"]) / f"{hemisphere}.pial.ras.ply").unlink()
    _assert_unavailable(context, "assets_unavailable")


@pytest.mark.parametrize("qc_status", ["provisional", "rejected"])
def test_unapproved_qc_preserves_provenance_but_blocks_download(cortex_cases, qc_status):
    context = cortex_cases
    context.entry_a["qc"]["status"] = qc_status
    context.entry_a["qc"]["notes"] = "Research reconstruction; not clinically validated."
    _write_registry(context)
    status = _assert_unavailable(context, "qc_not_approved")
    assert status["reconstruction_id"] == "trial-a"
    assert status["qc"] == context.entry_a["qc"]
    assert status["qc"]["clinically_validated"] is False
    assert _status(context, context.case_b)["available"] is True


@pytest.mark.parametrize("mode", ["disabled", "none", "missing", "malformed"])
def test_registry_unavailability_preserves_standard_viewer(cortex_cases, monkeypatch, mode):
    context = cortex_cases
    if mode == "disabled":
        context.registry["enabled"] = False
        _write_registry(context)
        reason = "registry_disabled"
    elif mode == "none":
        monkeypatch.setattr(paths, "CORTEX_REGISTRY_PATH", None)
        reason = "registry_disabled"
    elif mode == "missing":
        context.registry_path.unlink()
        reason = "registry_missing"
    else:
        context.registry_path.write_text("{invalid JSON", encoding="utf-8")
        reason = "registry_invalid"
    _assert_unavailable(context, reason)
    base = f"/api/v1/cases/{context.case_a}/viewer"
    assert context.client.get(f"{base}/metadata").status_code == 200
    assert context.client.get(f"{base}/slices/t1n/axial/2").status_code == 200


@pytest.mark.parametrize("field", ["reconstruction_id", "source_t1n_sha256"])
def test_duplicate_registry_identity_fails_closed(cortex_cases, field):
    context = cortex_cases
    context.entry_b[field] = context.entry_a[field]
    _write_registry(context)
    _assert_unavailable(context, "registry_invalid")


@pytest.mark.parametrize("field,value", [
    ("coordinate_convention", "voxel XYZ"),
    ("units", "metres"),
    ("source_t1n_sha256", "invalid"),
    ("qc", {"status": "unknown"}),
    ("qc", {
        "status": "approved", "scope": "research",
        "clinically_validated": True, "notes": "Unsupported clinical claim.",
    }),
])
def test_invalid_registry_metadata_fails_closed(cortex_cases, field, value):
    context = cortex_cases
    context.entry_a[field] = value
    _write_registry(context)
    _assert_unavailable(context, "registry_invalid")


@pytest.mark.parametrize("hemisphere", ["left", "LH", "unknown", "..", "%2e%2e%2fsecret"])
def test_invalid_hemisphere_never_serves_assets(cortex_cases, hemisphere):
    response = _mesh(cortex_cases, hemisphere)
    assert response.status_code in {400, 404}
    if hemisphere in {"left", "LH", "unknown"}:
        assert response.status_code == 400
    assert not response.content.startswith(b"ply\n")


@pytest.mark.parametrize("mesh_path", ["../outside.ply", "/tmp/outside.ply"])
def test_registry_asset_path_traversal_fails_closed(cortex_cases, mesh_path):
    context = cortex_cases
    context.entry_a["meshes"]["lh"]["path"] = mesh_path
    _write_registry(context)
    _assert_unavailable(context, "registry_invalid")


@pytest.mark.parametrize("target", ["mesh", "asset_root", "source", "modalities"])
def test_symlinks_cannot_escape_asset_or_staged_source_roots(cortex_cases, target):
    context = cortex_cases
    if target == "mesh":
        link = Path(context.entry_a["asset_root"]) / "lh.pial.ras.ply"
        destination = context.registry_path.parent / "outside.ply"
    elif target == "asset_root":
        link = Path(context.entry_a["asset_root"])
        destination = link.with_name("outside-assets")
    elif target == "source":
        link = context.live_cases / context.case_a / "modalities" / "t1n.nii.gz"
        destination = context.registry_path.parent / "outside.nii.gz"
    else:
        link = context.live_cases / context.case_a / "modalities"
        destination = context.registry_path.parent / "outside-modalities"
    is_directory = link.is_dir()
    link.rename(destination)
    link.symlink_to(destination, target_is_directory=is_directory)
    status = _status(context)
    assert status["available"] is False
    assert status["reason"] in {"assets_unavailable", "source_unavailable", "registry_invalid"}
    assert _mesh(context).status_code == 404


def test_missing_or_unready_staged_source_disables_cortex(cortex_cases):
    context = cortex_cases
    source = context.live_cases / context.case_a / "modalities" / "t1n.nii.gz"
    source.unlink()
    _assert_unavailable(context, "source_unavailable")
    source.write_bytes(context.source_a)
    metadata = context.live_cases / context.case_a / "case.json"
    case = json.loads(metadata.read_text(encoding="utf-8"))
    case["ready_for_inference"] = False
    metadata.write_text(json.dumps(case), encoding="utf-8")
    _assert_unavailable(context, "case_not_ready")


def test_invalid_or_absent_case_uses_existing_m11_errors(cortex_cases):
    context = cortex_cases
    for case_id, expected in (("invalid", 400), ("case_" + "f" * 32, 404)):
        base = f"/api/v1/cases/{case_id}/viewer/cortex"
        assert context.client.get(f"{base}/status").status_code == expected
        assert context.client.get(f"{base}/lh").status_code == expected


def test_registry_changes_take_effect_without_stale_qc_approval(cortex_cases):
    context = cortex_cases
    assert _mesh(context).status_code == 200
    context.entry_a["qc"]["status"] = "provisional"
    _write_registry(context)
    _assert_unavailable(context, "qc_not_approved")


def test_missing_assets_for_another_entry_do_not_disable_matching_case(cortex_cases):
    context = cortex_cases
    (Path(context.entry_b["asset_root"]) / "rh.pial.ras.ply").unlink()
    assert _status(context)["available"] is True
    assert _mesh(context).status_code == 200
    assert _status(context, context.case_b)["available"] is False


def test_normalized_hemisphere_aliases_are_not_separate_meshes(cortex_cases):
    context = cortex_cases
    context.entry_a["meshes"]["rh"] = {
        **context.entry_a["meshes"]["lh"], "path": "./lh.pial.ras.ply",
    }
    _write_registry(context)
    _assert_unavailable(context, "registry_invalid")


def test_storage_root_symlink_ancestor_is_rejected(cortex_cases):
    context = cortex_cases
    parent = context.registry_path.parent
    alias = parent / "storage-link"
    alias.symlink_to(parent, target_is_directory=True)
    context.entry_a["asset_root"] = str(alias / "assets-a")
    _write_registry(context)
    _assert_unavailable(context, "assets_unavailable")


def test_cortex_reads_preserve_source_metadata_registry_and_assets(cortex_cases):
    context = cortex_cases
    files = [context.registry_path]
    files.extend((context.live_cases / context.case_a).rglob("*"))
    files.extend(Path(context.entry_a["asset_root"]).iterdir())
    before = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in files if path.is_file()
    }
    assert _status(context)["available"] is True
    assert _mesh(context, "lh").status_code == 200
    assert _mesh(context, "rh").status_code == 200
    for path, snapshot in before.items():
        assert (path.stat().st_mtime_ns, path.read_bytes()) == snapshot

# M11E6B7G3.3 — Contour HTTP regression tests

def _install_binary_contour_test_meshes(context):
    """Replace temporary ASCII meshes with synthetic binary geometry."""
    import struct

    for hemisphere, shift_x in (("lh", 0.0), ("rh", 3.0)):
        # Synthetic voxel-space triangle crossing all three planes.
        voxel_vertices = np.array(
            [
                [0.25 + shift_x, 0.5, -1.0],
                [2.00 + shift_x, 4.0, 3.0],
                [0.25 + shift_x, 0.5, 3.0],
            ],
            dtype=np.float32,
        )

        # Match the synthetic NIfTI affine in _nifti_bytes().
        world_vertices = (
            voxel_vertices
            * np.array([1.25, 1.5, 2.0], dtype=np.float32)
        ).astype("<f4")

        header = (
            b"ply\n"
            b"format binary_little_endian 1.0\n"
            b"comment coordinates physical_RAS_plus_mm\n"
            b"element vertex 3\n"
            b"property float x\n"
            b"property float y\n"
            b"property float z\n"
            b"element face 1\n"
            b"property list uchar int vertex_indices\n"
            b"end_header\n"
        )

        payload = (
            header
            + world_vertices.tobytes()
            + struct.pack("<Biii", 3, 0, 1, 2)
        )

        filename = f"{hemisphere}.pial.ras.ply"
        root = Path(context.entry_a["asset_root"])
        (root / filename).write_bytes(payload)

        context.entry_a["meshes"][hemisphere] = {
            "path": filename,
            "sha256": _digest(payload),
        }

    _write_registry(context)


def _contour_response(context, plane="axial", index=1):
    return context.client.get(
        f"/api/v1/cases/{context.case_a}"
        f"/viewer/cortex/contours/{plane}/{index}"
    )


@pytest.mark.parametrize(
    ("plane", "index", "size"),
    [
        ("axial", 1, (6, 5)),
        ("coronal", 2, (6, 4)),
        ("sagittal", 1, (5, 4)),
        ("sagittal", 4, (5, 4)),
    ],
)
def test_approved_contour_http_png(
    cortex_cases, plane, index, size
):
    from io import BytesIO
    from PIL import Image

    context = cortex_cases
    _install_binary_contour_test_meshes(context)

    response = _contour_response(context, plane, index)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")

    image = Image.open(BytesIO(response.content))
    assert image.mode == "RGBA"
    assert image.size == size

    pixels = np.asarray(image)
    assert np.any(pixels[:, :, 3] > 0)
    assert np.any(pixels[:, :, 3] == 0)


@pytest.mark.parametrize("qc_status", ["provisional", "rejected"])
def test_contours_denied_for_unapproved_qc(
    cortex_cases, qc_status
):
    context = cortex_cases
    _install_binary_contour_test_meshes(context)

    context.entry_a["qc"]["status"] = qc_status
    _write_registry(context)

    response = _contour_response(context)

    assert response.status_code == 404
    assert _status(context)["reason"] == "qc_not_approved"


def test_contours_denied_after_source_mutation(cortex_cases):
    context = cortex_cases
    _install_binary_contour_test_meshes(context)

    staged_t1n = (
        context.live_cases
        / context.case_a
        / "modalities"
        / "t1n.nii.gz"
    )
    assert staged_t1n.is_file()

    # Modify the temporary staged source AFTER validation.
    staged_t1n.write_bytes(context.source_b)

    response = _contour_response(context)

    assert response.status_code == 404
    assert _status(context)["reason"] == "source_identity_mismatch"


def test_contours_denied_for_corrupted_mesh(cortex_cases):
    context = cortex_cases
    _install_binary_contour_test_meshes(context)

    mesh = (
        Path(context.entry_a["asset_root"])
        / "lh.pial.ras.ply"
    )
    mesh.write_bytes(mesh.read_bytes() + b"tampered")

    response = _contour_response(context)

    assert response.status_code == 404
    assert _status(context)["reason"] == "asset_integrity_failed"


@pytest.mark.parametrize(
    ("plane", "index"),
    [
        ("oblique", 1),
        ("axial", 99),
        ("sagittal", -1),
    ],
)
def test_contour_http_rejects_invalid_slice(
    cortex_cases, plane, index
):
    context = cortex_cases
    _install_binary_contour_test_meshes(context)

    response = _contour_response(context, plane, index)

    assert response.status_code == 400
