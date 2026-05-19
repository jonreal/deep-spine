#!/bin/bash

# Compare spine (ia_reflex) vs end-to-end (identity) with optimal HPs
# Each config has its own optimal params hardcoded:
#   - eye (identity): gs=2, lr=0.0003
#   - ia_reflex (spine): gs=8, lr=0.0001

LOG_DIR="./logs/rl_mass/run1"

mkdir -p "$LOG_DIR"

export LOG_DIR

format_custom_json() {
    local log_dir="$1"
    local seed="$2"
    cat <<EOF
{
  "log_dir": "$log_dir",
  "options": {
    "seed": $seed,
    "check_points": 10,
    "total_sim_steps": 5000000
  }
}
EOF
}

export -f format_custom_json

parallel --delay 2 --ungroup --tag --jobs 10 '
  source ../../myenv/bin/activate &&
  JSON=$(format_custom_json "$LOG_DIR" {2}) &&
  python run.py \
    --config {1} \
    --custom "$JSON"
' ::: p-pend-sac-eye.json p-pend-sac-ia_reflex.json ::: 1 2 3 4 5
