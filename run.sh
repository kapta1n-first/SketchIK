#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Reuse installed third-party dependencies only; no old project package is imported.
runtime_file="$project_root/.runtime/python"
if [[ -n "${SKETCHIK_PYTHON:-}" ]]; then
    runtime_python="$SKETCHIK_PYTHON"
elif [[ -f "$runtime_file" ]]; then
    IFS= read -r runtime_python < "$runtime_file"
else
    runtime_python="$project_root/.venv/bin/python"
fi
if [[ ! -x "$runtime_python" ]]; then
    echo 'Set SKETCHIK_PYTHON to a Python with LIBERO, MuJoCo and robosuite installed.' >&2
    exit 1
fi
export PYTHONPATH="$project_root/src"
export PYTHONDONTWRITEBYTECODE=1
exec "$runtime_python" -m sketchik.cli "$@"
