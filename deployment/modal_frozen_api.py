"""NeuroPrompt-3D: first genuine Modal research API.

Uses the original immutable M9E6 evaluation service.
No synthetic records or regenerated metrics.
"""

import modal

app = modal.App("neuroprompt3d-real-backend")

EVIDENCE = (
    "archives/M9E/2026-09-29/"
    "M9E6_FINAL_TEST_EVALUATION/"
    "M9E6_FINAL_TEST_EVALUATION_2026_09_29.json"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.142.2",
        "numpy==2.4.6",
        "nibabel==5.4.2",
        "pillow==12.0.0",
    )
    .add_local_file(
        "app/backend/core/paths.py",
        "/root/app/backend/core/paths.py",
    )
    .add_local_file(
        "app/backend/services/frozen_test_service.py",
        "/root/app/backend/services/frozen_test_service.py",
    )
    .add_local_file(
        EVIDENCE,
        "/root/" + EVIDENCE,
    )
)


for service in (
    "frozen_display_contrast",
    "frozen_slice_service",
    "frozen_initial_position_service",
    "frozen_removed_focus_service",
):
    image = image.add_local_file(
        f"app/backend/services/{service}.py",
        f"/root/app/backend/services/{service}.py",
    )

image = image.add_local_dir(
    "archives/M9E",
    "/root/archives/M9E",
)


@app.function(
    image=image,
    cpu=0.25,
    memory=1024,
    max_containers=1,
)
@modal.asgi_app()
def research_api():
    from fastapi import FastAPI, Query

    from app.backend.services.frozen_test_service import (
        get_frozen_test_summary,
        get_frozen_test_cases,
        get_frozen_test_case,
    )

    from app.backend.services.frozen_initial_position_service import (
        get_verified_initial_position,
    )
    from app.backend.services.frozen_removed_focus_service import (
        get_verified_removed_focus,
    )

    api = FastAPI(
        title="NeuroPrompt-3D Verified Research API",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @api.get("/api/v1/frozen-test/summary")
    def summary():
        return get_frozen_test_summary()

    @api.get("/api/v1/frozen-test/cases")
    def cases(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=20, ge=1, le=129),
        action: str | None = None,
    ):
        return get_frozen_test_cases(
            offset=offset,
            limit=limit,
            action=action,
        )

    @api.get("/api/v1/frozen-test/cases/{case_index}")
    def case(case_index: int):
        return get_frozen_test_case(case_index=case_index)

    @api.get("/api/v1/frozen-test/cases/{case_index}/initial-position")
    def initial_position(case_index: int):
        return get_verified_initial_position(case_index)

    @api.get("/api/v1/frozen-test/cases/{case_index}/removed-focus")
    def removed_focus(case_index: int):
        return get_verified_removed_focus(case_index)

    return api
