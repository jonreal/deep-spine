#!/bin/bash

LOG_DIR="./logs/PNAS/m0-b4-sweeps_S1_E1_B025-P1D2/run1"

mkdir -p "$LOG_DIR"
cp "$0" "$LOG_DIR/launch-$(date +%Y%m%d-%H%M%S).sh.txt"
export LOG_DIR

format_custom_json() {
    local log_dir="$1"
    local strength="$2"
    local babble="$3"
    local gain="$4"
    cat <<EOF
{
  "log_dir": "$log_dir",
  "options": {
    "num_eps": 2000,
    "max_muscle_strength": $strength
  },
  "env_kwargs": {
    "body_mass": [0, 1, 2, 0],
    "relative_muscle_scale": [1, 1, 0.5, 0.5, 0.25, 0.25]
  },
  "supraspine_kwargs": {
    "babble_max_prob": $babble
  },
  "hebbian_kwargs": [
    {
      "gain": $gain
    }
  ]
}
EOF
}

export -f format_custom_json

#parallel --delay 5 --line-buffer --tag --jobs 15 '
#  source ../../myenv/bin/activate &&
#  JSON=$(format_custom_json "$LOG_DIR" {1} {2} {3}) &&
#  python run.py \
#    --config p-babble.json \
#    --custom "$JSON"
#' ::: 0.9 1 ::: 0.0003 0.0007 0.0018 0.0042 0.0100 ::: 3.5

#### pb = logspace(-3.5, -0.125, 10)
# pb =
#    0.0003 0.0007 0.0018 0.0042 0.0100 0.0237 0.0562 0.1334 0.3162  0.7499

parallel --delay 10 --line-buffer --tag --jobs 20 '
  source ../../myenv/bin/activate &&
  JSON=$(format_custom_json "$LOG_DIR" {1} {2} {3}) &&
  python run.py \
    --config p-babble.json \
    --custom "$JSON"
' ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 ::: 0.0003 0.0007 0.0018 0.0042 0.0100 0.0237 0.0562 ::: 3.5

