#!/bin/bash

LOG_DIR="./logs/PNAS-prep/"


mkdir -p "$LOG_DIR"
export LOG_DIR  # make it available to subshells
export PARAMS

format_custom_json() {
    local log_dir="$1"
    local seed="$2"
    cat <<EOF
{
  "log_dir": "$log_dir",
  "options": {
    "seed": "$seed"
  }
}
EOF
}

export -f format_custom_json

parallel --delay 2 --ungroup --tag --jobs 10 '
  source ../../myenv/bin/activate &&
  JSON=$(format_custom_json "$LOG_DIR" {2}) &&
  python train.py \
    --config {1} \
    --custom "$JSON"
' ::: parameters-pendulum-sac.json parameters-pendulum-sac-ia_reflex.json ::: 1 2 3 4 5

#parallel --delay 2 --line-buffer --tag --jobs 20 '
#  source ../../myenv/bin/activate &&
#  JSON=$(format_custom_json "$LOG_DIR" {1} {2} {3}) &&
#  python train.py \
#    --config parameters.json \
#    --custom "$JSON"
#' ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 ::: 0.001 0.0025 0.005 0.01 0.025 0.05 0.1 ::: 0.5 1 1.5 2 2.5

#parallel --delay 2 --line-buffer --tag --jobs 20 '
#  source ../../myenv/bin/activate &&
#  JSON=$(format_custom_json "$LOG_DIR" {1} {2} {3}) &&
#  python train.py \
#    --config parameters.json \
#    --custom "$JSON"
#' ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 ::: 0.0001 0.00025 0.0005 0.00075 0.0075 ::: 0.5 1 1.5 2 2.5

#parallel --delay 2 --line-buffer --tag --jobs 25 '
#  source ../../myenv/bin/activate &&
#  JSON=$(format_custom_json "$LOG_DIR" {1} {2} {3}) &&
#  python train.py \
#    --config parameters.json \
#    --custom "$JSON"
#' ::: 0.1 0.2 0.3 0.4 0.5 0.6 ::: 0.0003 0.0007 0.0018 0.0042 0.0100 ::: 2 2.5 3 3.5 4 4.5 5
#





