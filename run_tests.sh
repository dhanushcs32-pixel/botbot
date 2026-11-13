#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_tests.sh - run the offline regression tests added by patch 0008.
#
#   ./run_tests.sh /c/botbot
#   PYTHON=/c/botbot/venv/Scripts/python.exe ./run_tests.sh /c/botbot
#
# No NPU server, no chromadb, no model files are needed.
# Picks the repo's venv python automatically if one exists.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${1:-.}"
cd "$REPO"
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

[[ -f tests/test_safety_and_grounding.py ]] || { echo "tests/test_safety_and_grounding.py not found - apply the patches first (./apply.sh)."; exit 1; }

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x venv/Scripts/python.exe ]]; then   # Windows venv (your C:\botbot layout)
  PY="venv/Scripts/python.exe"
elif [[ -x venv/bin/python ]]; then
  PY="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

echo "Using: $PY"
"$PY" tests/test_safety_and_grounding.py 2>&1 | grep -v -E '^\[(laya|style_check)'
exit "${PIPESTATUS[0]}"

