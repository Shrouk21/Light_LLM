#!/usr/bin/env bash
# setup.sh - create .venv and install the exact package set this project was verified with.
# Needs uv (https://docs.astral.sh/uv/) and Python 3.12. Safe to re-run.
#   UV_LOCK_TIMEOUT  raise this if the install dies with "Failed to acquire lock"
#                    (another user's uv run is holding the shared cache)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

[ -x "$ROOT/.venv/bin/python" ] || uv venv --python 3.12 "$ROOT/.venv"
export UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-1200}"
uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.lock.txt" 2>&1 | tee -a "$ROOT/logs/install.log"

"$ROOT/.venv/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda ok:', torch.cuda.is_available())"
