#!/bin/bash

LOG_DIR="./logs/exp9/sweep_strength_babble_gain_0"
mkdir -p "$LOG_DIR"
export LOG_DIR  # make it available to subshells

format_custom_json() {
    local log_dir="$1"
    local strength="$2"
    local babble="$3"
    local gain="$4"
    cat <<EOF
{
  "log_dir": "$log_dir",
  "options": {
    "max_muscle_strength": $strength
  },
  "supraspine_kwargs": {
    "babble_max_prob": $babble
  },
  "hebbian_kwargs": [
    {},
    {
      "gain": $gain
    }
  ]
}
EOF
}

export -f format_custom_json

#parallel --delay 2 --line-buffer --tag --jobs 20 '
#  source ../../myenv/bin/activate &&
#  JSON=$(format_custom_json "$LOG_DIR" {1} {2} {3}) &&
#  python run.py \
#    --config parameters.json \
#    --custom "$JSON"
#' ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 ::: 0.001 0.0025 0.005 0.01 0.025 0.05 0.1 ::: 0.5 1 1.5 2 2.5

#parallel --delay 2 --line-buffer --tag --jobs 20 '
#  source ../../myenv/bin/activate &&
#  JSON=$(format_custom_json "$LOG_DIR" {1} {2} {3}) &&
#  python run.py \
#    --config parameters.json \
#    --custom "$JSON"
#' ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 ::: 0.0001 0.00025 0.0005 0.00075 0.0075 ::: 0.5 1 1.5 2 2.5

#parallel --delay 2 --line-buffer --tag --jobs 25 '
#  source ../../myenv/bin/activate &&
#  JSON=$(format_custom_json "$LOG_DIR" {1} {2} {3}) &&
#  python run.py \
#    --config parameters.json \
#    --custom "$JSON"
#' ::: 0.1 0.2 0.3 0.4 0.5 0.6 ::: 0.0003 0.0007 0.0018 0.0042 0.0100 ::: 2 2.5 3 3.5 4 4.5 5
#

parallel --delay 2 --line-buffer --tag --jobs 25 '
  source ../../myenv/bin/activate &&
  JSON=$(format_custom_json "$LOG_DIR" {1} {2} {3}) &&
  python run.py \
    --config parameters.json \
    --custom "$JSON"
' ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 ::: 0.0003 0.0007 0.0018 0.0042 0.0100 ::: 2 2.5 3 3.5 4 4.5 5



