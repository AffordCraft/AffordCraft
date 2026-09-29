#!/bin/bash
# Launch (or restart) the GPT-6 Astra agent-baseline run on the registered 200-input subset. Restart-safe: finished cases
# are skipped, half-done case directories are moved to interrupted/ and redone. One GPU serves the catalog encoder and the
# offscreen renderer. usage: run.sh
# environment: AGENT_RUN (run root), AGENT_CONCURRENCY (cases in flight; 16 at launch, 32 when the run was resumed),
# AGENT_GPU, AFFORDCRAFT_RUNTIME_PYTHON, EVAL_API_BASE / EVAL_API_KEY and the locations listed in common.py
set -u
ROOT=${AGENT_RUN:-${AFFORDCRAFT_RUNS:-runs}/external/gpt6-astra-agent/subset200}
H=$(cd "$(dirname "$0")" && pwd)
CONC=${AGENT_CONCURRENCY:-16}
: "${EVAL_API_BASE:?set EVAL_API_BASE}" "${EVAL_API_KEY:?set EVAL_API_KEY}"
if pgrep -f "run_lanes.py --root $ROOT" >/dev/null; then echo "controller already running"; exit 1; fi
cd $H && CUDA_VISIBLE_DEVICES=${AGENT_GPU:-0} EGL_DEVICE_ID=${AGENT_GPU:-0} OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 ${AFFORDCRAFT_RUNTIME_PYTHON:-python} run_lanes.py --root $ROOT --lanes 16 --concurrency $CONC
