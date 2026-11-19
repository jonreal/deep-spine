#!/bin/bash
# Canonical run (SR0-MR0, kappa 5, strength 0.2, babble 0.0046, seed 1) with
# per-step logs and video, then the per-step Hebbian update.
#
# The exact paper version of this launcher, which also cuts Movies S1-S5 with
# the paper's movie scripts, is in release v0.1.0 (see the README).
#
# The frame rate is set in a copy of configs/babble.json, not with --custom
# or --show: those print the overrides onto the video.
#
# Usage (repo root, environment active):
#   ./launch/canonical.sh
#   LOG_ROOT=./logs/verify ./launch/canonical.sh
#
# Output: $LOG_ROOT/<sim-id>/  (default LOG_ROOT=./logs/hebbian-canonical)
#   data.txt, env.txt, hebbian.txt     simulation and per-step logs (~21 GB)
#   dw_rule.bin                        per-step Hebbian update, Figs 3-4 (~5 GB)
#   sim.mp4                            full run at 10x real time (~20 GB)
#
# 2000 episodes = 50 h simulated; about 18 h wall clock with video. Stop it
# with Ctrl-C in this terminal (not under parallel) so the video is finalized.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

python -c "import deep_spine" 2>/dev/null || {
  echo "deep_spine not importable: activate the environment first (see README)" >&2
  exit 1
}

LOG_ROOT="${LOG_ROOT:-./logs/hebbian-canonical}"
mkdir -p "$LOG_ROOT"

CONFIG=$(mktemp "${TMPDIR:-/tmp}/canonical-video.XXXXXX")
trap 'rm -f "$CONFIG"' EXIT
python - "$CONFIG" "$LOG_ROOT" <<'EOF'
import json, sys
c = json.load(open('configs/babble.json'))
c['log_dir'] = sys.argv[2]
c['env_kwargs']['frame_rate'] = 20
json.dump(c, open(sys.argv[1], 'w'), indent=2)
EOF

python -m deep_spine --config "$CONFIG" --log

RUN="$LOG_ROOT/$(ls "$LOG_ROOT" | grep -E '^[0-9]+$' | sort -n | tail -1)"
echo "run: $RUN"

python analysis/export_dw_rule.py "$RUN"
# Paper movies (Movies S1-S5): scripts are in release v0.1.0, analysis/pubfigs/
# python analysis/pubfigs/cut_movies.py "$RUN/sim.mp4" "$RUN/si-clips"
# python analysis/pubfigs/timelapse_movie.py "$RUN/sim.mp4" "$RUN/si-clips/si-movie-full.mp4"
