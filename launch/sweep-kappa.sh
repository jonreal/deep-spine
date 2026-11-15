#!/bin/bash
# Hebbian gain (kappa) sweep: strength x babble grid (10 x 10) at kappa 1-4
# and 10, in the SR0-MR0 body. Kappa 5 is the SR0-MR0 embodiment sweep.
#
# Usage (repo root, environment active):
#   ./launch/sweep-kappa.sh                      # 500 runs
#   LOG_ROOT=./logs/verify ./launch/sweep-kappa.sh
#
# Output (folder names as stored):
#   $LOG_ROOT/SR0-MR0-gain/<sim-id>/      kappa 1-4
#   $LOG_ROOT/SR0-MR0-gain-10/<sim-id>/   kappa 10
# (default LOG_ROOT=./logs/hebbian-sweep)
# Jobs: JOBS (default: one per core, minus one).
#
# Never Ctrl-C under parallel: it sends SIGTERM, which skips env.close().
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

LOG_ROOT="${LOG_ROOT:-./logs/hebbian-sweep}"
JOBS="${JOBS:--1}"

# folder:kappa
RUNS=(SR0-MR0-gain:1 SR0-MR0-gain:2 SR0-MR0-gain:3 SR0-MR0-gain:4 SR0-MR0-gain-10:10)

# Same keys as the stored custom.txt files: analysis/extract_parameters.m builds
# the sweep table from them.
format_custom_json() {
    local log_dir="$1" str="$2" bab="$3" gain="$4"
    cat <<EOF
{
  "log_dir": "$log_dir",
  "options": {
    "num_eps": 2000,
    "max_muscle_strength": $str,
    "ramp_fraction": 0.5,
    "strength_shape": "linear"
  },
  "env_kwargs": {
    "viscosity": 0.01,
    "body_mass": [0, 2, 1, 0],
    "muscle_strength": 0.01,
    "relative_muscle_scale": [1, 1, 0.5, 0.5, 0.25, 0.25]
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

echo "log root    : $LOG_ROOT"
echo "grid        : ${#STRENGTH[@]} strength x ${#BABBLE[@]} babble per kappa"
for r in "${RUNS[@]}"; do
  printf '  %-16s  kappa=%s\n' "${r%%:*}" "${r##*:}"
  mkdir -p "$LOG_ROOT/${r%%:*}"
done
for d in $(printf '%s\n' "${RUNS[@]%%:*}" | sort -u); do
  cp "$0" "$LOG_ROOT/$d/launch-$(date +%Y%m%d-%H%M%S).sh.txt"
done
echo "total runs  : $(( ${#RUNS[@]} * ${#STRENGTH[@]} * ${#BABBLE[@]} ))"
echo

export LOG_ROOT
export -f format_custom_json

parallel --will-cite --delay 10 --line-buffer --tag --jobs "$JOBS" --colsep : '
  JSON=$(format_custom_json "$LOG_ROOT/{1}" {3} {4} {2}) &&
  python -m deep_spine --config configs/canonical.json --custom "$JSON"
' ::: "${RUNS[@]}" ::: "${STRENGTH[@]}" ::: "${BABBLE[@]}"
