#!/bin/bash

LOG_DIR="./logs/rl-explore-5/compare-sweep_mass_8"

mkdir -p "$LOG_DIR"
cp "$0" "$LOG_DIR/launch-$(date +%Y%m%d-%H%M%S).sh.txt"
export LOG_DIR

format_custom_json() {
    local log_dir="$1"
    local df="$2"
    local lr="$3"
    local gs="$4"
    local td="$5"
    local na="$6"
    local ms="$7"
    local pm="$8"
    local seed="$9"

    cat <<EOF
{
  "log_dir": "$log_dir",
  "options": {
    "seed": $seed,
    "check_points": 100,
    "total_sim_steps": 15000000,
    "eval_eps": 50
  },
  "env_kwargs": {
     "muscle_strength": $ms,
     "reward_type": "gauss",
     "body_mass": [0, 2, 1, $pm, 0],
     "damping": [0, 0, 1]
  },
  "supraspine_kwargs": {
    "decimation_factor": $df,
    "tap_delays": $td,
    "kwargs": {
      "learning_rate": $lr,
      "gradient_steps": $gs,
      "net_arch":[$na,$na]
    }
  }
}
EOF
}

export -f format_custom_json

parallel --delay 2 --ungroup --tag --jobs 15 '
  source ../../myenv/bin/activate &&
  JSON=$(format_custom_json "$LOG_DIR" {2} {3} {4} {5} {6} {7} {8} {9}) &&
  python run.py \
    --config {1} \
    --custom "$JSON" \
  ' ::: no-spine.json reflex-diag.json \
    ::: 40 \
    ::: 1e-3 \
    ::: 1 2 \
    ::: 16 \
    ::: 32 \
    ::: 1 \
    ::: 0.5 1 2 \
    ::: 1 2 3

#  ' ::: ref-diag.json \
#    ::: 20 \
#    ::: 1e-3 \
#    ::: 1 2 4 \
#    ::: 0 1 2 4 8 16 ::: 32 \
#    ::: 1 2 3
#
#
#  ' ::: no-spine.json reflex-diag.json reflex-syn.json no-spine-odd.json reflex-odd.json ::: 1e-3 3e-3 ::: 2 4 ::: 1 2 3
