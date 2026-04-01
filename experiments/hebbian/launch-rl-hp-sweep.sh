#!/bin/bash

LOG_DIR="./logs/rl-explore/compare"

mkdir -p "$LOG_DIR"
export LOG_DIR

format_custom_json() {
    local log_dir="$1"
    local lr="$2"
    local gs="$3"
    local seed="$4"
    local mass="$5"
    local damp="$6"

    cat <<EOF
{
  "log_dir": "$log_dir",
  "options": {
    "seed": $seed,
    "check_points": 100,
    "total_timesteps": 500000
  },
  "env_kwargs": {
    "reward_type": "simple",
    "damping": $damp,
    "body_mass":[0,2,1,$mass,0],
    "pendulum_length": 0.5,
    "relative_muscle_scale": [1, 1, 0.5, 0.5, 0.25, 0.25],
    "muscle_strength": 1
  },
  "supraspine_kwargs": {
    "visual_mode": "joint",
    "kwargs": {
      "learning_rate": $lr,
      "gradient_steps": $gs
    }
  }
}
EOF
}

export -f format_custom_json

parallel --delay 2 --ungroup --tag --jobs 10 '
  source ../../myenv/bin/activate &&
  JSON=$(format_custom_json "$LOG_DIR" {2} {3} {4} {5} {6}) &&
  python run.py \
    --config {1} \
    --custom "$JSON"
  ' ::: p-pend-sac-eye.json p-pend-sac-ia_reflex.json ::: 1e-4 ::: 8 ::: 1 2 3 ::: 0.25 0.5 1 ::: 1 0.01
