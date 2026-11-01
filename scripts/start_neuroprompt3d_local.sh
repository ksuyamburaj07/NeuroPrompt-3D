#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$PROJECT_ROOT"

if [[ ! -f app/frontend/dist/index.html ]]; then
    echo "STOP: Production frontend missing."
    echo "Build using: cd app/frontend && npm run build"
    exit 1
fi

if ! python -c 'import uvicorn' >/dev/null 2>&1; then
    echo "STOP: Activate conda environment neuroprompt3d_m11."
    exit 1
fi

echo "============================================"
echo " NeuroPrompt-3D — Unified Local Application "
echo "============================================"
echo
echo "Open: http://127.0.0.1:8090"
echo "Local-only research prototype."
echo "Press Ctrl+C to shut down."
echo

exec python -m uvicorn app.backend.local_unified:app \
  --host 127.0.0.1 \
  --port 8090 \
  --workers 1
