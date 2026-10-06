from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.api.cases import router as cases_router
from app.backend.api.health import router as health_router
from app.backend.api.runs import router as runs_router
from app.backend.core.constants import (
    API_PREFIX,
    APP_NAME,
    APP_VERSION,
    RESEARCH_DISCLAIMER,
)
from app.backend.core.paths import ensure_runtime_directories


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    ensure_runtime_directories()
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description=RESEARCH_DISCLAIMER,
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(
        health_router,
        prefix=API_PREFIX,
    )

    application.include_router(
        cases_router,
        prefix=API_PREFIX,
    )

    application.include_router(
        runs_router,
        prefix=API_PREFIX,
    )

    return application


app = create_app()
