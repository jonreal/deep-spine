#!/bin/bash
# Embodiment sweep: strength x babble grid (10 x 10) for each body.
#
#   SR = strength ratio, relative_muscle_scale over [SF SE EF EE BF BE]
#   MR = mass ratio, body_mass over [ground, upper arm, forearm, hand]
#
# Each run is configs/canonical.json with the overrides below; the canonical
# run is the SR0-MR0 point at strength 0.2, babble 0.0046.
#
# Usage (repo root, environment active):
#   ./launch/sweep-embodiment.sh                  # all six bodies, 600 runs
#   ./launch/sweep-embodiment.sh SR2-MR0          # one body, 100 runs
#   LOG_ROOT=./logs/verify ./launch/sweep-embodiment.sh SR0-MR0
#
# Output: $LOG_ROOT/<tag>/<sim-id>/  (default LOG_ROOT=./logs/hebbian-sweep)
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
TAGS=("$@")
if [ ${#TAGS[@]} -eq 0 ]; then
  TAGS=(SR0-MR0 SR0-MR1 SR1-MR0 SR1-MR1 SR2-MR0 SR2-MR1)
fi

rel_scale() {
  case "$1" in
    SR0-*) echo '[1, 1, 0.5, 0.5, 0.25, 0.25]'     ;;  # 1:0.5:0.25
    SR1-*) echo '[1, 1, 0.25, 0.25, 0.5, 0.5]'     ;;  # 1:0.25:0.5
    SR2-*) echo '[0.5, 0.5, 0.5, 0.5, 0.75, 0.75]' ;;  # 0.5:0.5:0.75 (mutant)
    *) echo "unknown SR in tag '$1'" >&2; return 1 ;;
  esac
}

# Total mass is 3.0 in both bodies; only the proximal:distal ratio changes.
body_mass() {
  case "$1" in
    *-MR0) echo '[0, 2, 1, 0]'     ;;  # 2:1
    *-MR1) echo '[0, 1.5, 1.5, 0]' ;;  # 1.5:1.5
    *) echo "unknown MR in tag '$1'" >&2; return 1 ;;
  esac
}

# Same keys as the stored custom.txt files: analysis/extract_parameters.m builds
# the sweep table from them.
format_custom_json() {
    local log_dir="$1" str="$2" bab="$3" rms="$4" mass="$5"
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
    "body_mass": $mass,
    "muscle_strength": 0.01,
    "relative_muscle_scale": $rms
  },
  "supraspine_kwargs": {
    "babble_max_prob": $bab
  },
  "hebbian_kwargs": [
    {
      "gain": 5,
      "learning_rate": 0.1
    }
  ]
}
EOF
}

STRENGTH=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
BABBLE=(0.0010 0.0017 0.0028 0.0046 0.0077 0.0129 0.0215 0.0359 0.0599 0.1000)

echo "log root    : $LOG_ROOT"
echo "grid        : ${#STRENGTH[@]} strength x ${#BABBLE[@]} babble per body"
for t in "${TAGS[@]}"; do
  rms=$(rel_scale "$t")
  mass=$(body_mass "$t")
  printf '  %-9s  relative_muscle_scale=%-32s  body_mass=%s\n' "$t" "$rms" "$mass"
  mkdir -p "$LOG_ROOT/$t"
  cp "$0" "$LOG_ROOT/$t/launch-$(date +%Y%m%d-%H%M%S).sh.txt"
done
echo "total runs  : $(( ${#TAGS[@]} * ${#STRENGTH[@]} * ${#BABBLE[@]} ))"
echo

export LOG_ROOT
export -f format_custom_json rel_scale body_mass

parallel --will-cite --delay 10 --line-buffer --tag --jobs "$JOBS" '
  JSON=$(format_custom_json "$LOG_ROOT/{1}" {2} {3} "$(rel_scale {1})" "$(body_mass {1})") &&
  python -m deep_spine --config configs/canonical.json --custom "$JSON"
' ::: "${TAGS[@]}" ::: "${STRENGTH[@]}" ::: "${BABBLE[@]}"
