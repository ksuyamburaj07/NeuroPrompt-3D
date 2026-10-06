from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

APP_ROOT = PROJECT_ROOT / "app"
BACKEND_ROOT = APP_ROOT / "backend"
RUNTIME_ROOT = BACKEND_ROOT / "runtime"

LIVE_CASES_ROOT = RUNTIME_ROOT / "live_cases"
LIVE_RUNS_ROOT = RUNTIME_ROOT / "live_runs"

FROZEN_M9E_ROOT = PROJECT_ROOT / "archives" / "M9E"
FROZEN_M10_ROOT = PROJECT_ROOT / "archives" / "M10"

BASELINE_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "M7K_C_Full_Baseline_Training"
    / "best_baseline.pt"
)


def ensure_runtime_directories() -> None:
    LIVE_CASES_ROOT.mkdir(parents=True, exist_ok=True)
    LIVE_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
