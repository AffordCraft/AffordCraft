#!/usr/bin/env bash
# Round 2 (final): after the round-1 evaluation, collect dagger20r2 with the round-1 head (EP=6 episodes per worker and
# scene, seeds 2200+worker), build the round-2 dataset = teacher successes (collect20) + ALL DAgger episodes of dagger20
# and dagger20r2 (failed ones truncated 45 ticks before their end and kept only with >= 120 remaining ticks;
# object-off-table episodes rejected), train the round-2 head reusing the round-1 features, then evaluate round 2
# (held-out seed 2600: trained 10 / untrained 4 per scene; seen seed 2000: trained 6 per scene).
# Prerequisite: round 1 finished (run_dagger.sh -> post_train.sh 1 -> HEAD=.../round1 eval.sh).
set -uo pipefail
F=$(cd "$(dirname "$0")" && pwd)
S=${AFFORDCRAFT_RUNS:-runs}/scenes_vla
TAG=dagger20r2 HEAD=$S/v20_vla/round1 SEED0=2200 EP=${EP:-6} bash "$F/run_dagger.sh"
TAGS_R1="collect20 dagger20 dagger20r2" DS_EXTRA="--include-failed-modes dagger --failed-truncate-ticks 45 --failed-min-ticks 120" bash "$F/post_train.sh" 2
[ -f "$S/v20_vla/round2/completion.json" ] || { echo "round 2: training did not complete"; exit 1; }
HEAD=$S/v20_vla/round2 bash "$F/eval.sh"
