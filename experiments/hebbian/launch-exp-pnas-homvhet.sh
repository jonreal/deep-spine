#!/bin/bash

LOG_DIR="./logs/PNAS/homvhet_3"
mkdir -p "$LOG_DIR"
export LOG_DIR  # make it available to subshells

format_custom_json() {
    local log_dir="$1"
    local seed="$2"
    cat <<EOF
{
  "log_dir": "$log_dir",
  "options": {
    "seed": $seed
  }
}
EOF
}

export -f format_custom_json

parallel --delay 2 --line-buffer --tag --jobs 25 '
  source ../../myenv/bin/activate &&
  JSON=$(format_custom_json "$LOG_DIR" {1}) &&
  python train.py \
    --log \
    --config parameters-pnas-homvhet.json \
    --custom "$JSON"
' ::: 1 2

