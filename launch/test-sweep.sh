#!/bin/bash
# Quick end-to-end test of the sweep + analysis pipeline: the embodiment sweep
# grid (10 strength x 10 babble) for one body at two Hebbian gains, with only
# NUM_EPS episodes per run (default 10, ~15 simulated min).
#
# Usage (repo root, environment active):
#   ./launch/test-sweep.sh                    # SR0-MR0, kappa 2 and 5, 200 runs
#   TAG=SR2-MR0 KAPPAS="5" ./launch/test-sweep.sh
#
# Output: $LOG_ROOT/<tag>/<sim-id>/  (default LOG_ROOT=./logs/test-sweep)
# Then, in MATLAB (analysis/ on the path):
#   T = ds_process_babble('<repo>/logs/test-sweep/SR0-MR0', printfig=true, FigVis='off');
#   ds_plot_motif_heatmaps(T);  ds_plot_motif_gui(T, root='<repo>/logs/test-sweep/SR0-MR0');
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

command -v parallel >/dev/null || {
  echo "GNU parallel is required: brew install parallel (macOS) or apt install parallel (Linux)" >&2
  exit 1
}
python -c "import deep_spine" 2>/dev/null || {
  echo "deep_spine not importable: activate the environment first (see README)" >&2
  exit 1
}

LOG_ROOT="${LOG_ROOT:-./logs/test-sweep}"
TAG="${TAG:-SR0-MR0}"
KAPPAS=(${KAPPAS:-2 5})
NUM_EPS="${NUM_EPS:-10}"
JOBS="${JOBS:--1}"

case "$TAG" in
  SR0-*) RMS='[1, 1, 0.5, 0.5, 0.25, 0.25]'     ;;
  SR1-*) RMS='[1, 1, 0.25, 0.25, 0.5, 0.5]'     ;;
  SR2-*) RMS='[0.5, 0.5, 0.5, 0.5, 0.75, 0.75]' ;;
  *) echo "unknown SR in tag '$TAG'" >&2; exit 1 ;;
esac
case "$TAG" in
  *-MR0) MASS='[0, 2, 1, 0]'     ;;
  *-MR1) MASS='[0, 1.5, 1.5, 0]' ;;
  *) echo "unknown MR in tag '$TAG'" >&2; exit 1 ;;
esac

format_custom_json() {
    local log_dir="$1" str="$2" bab="$3" gain="$4"
    cat <<EOF
{
  "log_dir": "$log_dir",
  "options": {
    "num_eps": $NUM_EPS,
    "max_muscle_strength": $str,
    "ramp_fraction": 0.5,
    "strength_shape": "linear"
  },
  "env_kwargs": {
    "viscosity": 0.01,
    "body_mass": $MASS,
    "muscle_strength": 0.01,
    "relative_muscle_scale": $RMS
  },
  "supraspine_kwargs": {
    "babble_max_prob": $bab
  },
  "hebbian_kwargs": [
    {
      "gain": $gain,
      "learning_rate": 0.1
    }
  ]
}
EOF
}

STRENGTH=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
BABBLE=(0.0010 0.0017 0.0028 0.0046 0.0077 0.0129 0.0215 0.0359 0.0599 0.1000)

mkdir -p "$LOG_ROOT/$TAG"
echo "test sweep: $TAG, kappa ${KAPPAS[*]}, $NUM_EPS episodes, $(( ${#KAPPAS[@]} * ${#STRENGTH[@]} * ${#BABBLE[@]} )) runs -> $LOG_ROOT/$TAG"

export LOG_ROOT TAG NUM_EPS RMS MASS
export -f format_custom_json

# run jobs in bash: the exported functions do not exist in other shells, and
# parallel falls back to $SHELL when it cannot detect the calling shell
export PARALLEL_SHELL=bash

parallel --will-cite --delay 1 --line-buffer --jobs "$JOBS" '
  JSON=$(format_custom_json "$LOG_ROOT/$TAG" {2} {3} {1}) &&
  python -m deep_spine --config configs/babble.json --custom "$JSON"
' ::: "${KAPPAS[@]}" ::: "${STRENGTH[@]}" ::: "${BABBLE[@]}"
