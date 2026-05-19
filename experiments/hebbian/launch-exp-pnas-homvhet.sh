#!/bin/bash

LOG_DIR="./logs/PNAS/homvhet_11"
mkdir -p "$LOG_DIR"
export LOG_DIR  # make it available to subshells

format_custom_json() {
    local log_dir="$1"
    local s1="$2"
    local s2="$3"
    local s3="$4"
    cat <<EOF
{
  "log_dir": "$log_dir",
  "env_kwargs": {
    "relative_muscle_scale": [$s1, $s1, $s2, $s2, $s3, $s3]
  }
}
EOF
}

export -f format_custom_json

TRIPLETS=(
  "1 1 1"
  "1 0.5 0.25"
  "1 0.25 0.5"
  "0.5 0.5 0.75"
)

parallel --delay 2 --line-buffer --tag --jobs 25 "
  source ../../myenv/bin/activate &&
  read -r s1 s2 s3 <<< {1} &&
  JSON=\$(format_custom_json \"$LOG_DIR\" \"\$s1\" \"\$s2\" \"\$s3\") &&
  python run.py \
    --log \
    --config parameters-pnas-homvhet.json \
    --custom \"\$JSON\"
" ::: "${TRIPLETS[@]}"

