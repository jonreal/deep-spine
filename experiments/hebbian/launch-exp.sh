#!/bin/bash

# Array of values for lock_target
values=(1 2 3 4 5)

LOG_DIR="./logs/exp2/seed"

# Directory path
DIR="/Users/jonreal/research/deep-spine/experiments/hebbian"
MYENV_PATH="../../myenv/bin/activate"

# Create a unique session name using timestamp
SESSION_NAME="exp_$(date +%s)"

# Start a new tmux session
tmux new-session -d -s $SESSION_NAME

# For each value, create a pane and run the command
for ((i=0; i<${#values[@]}; i++)); do
    # For the first value, use the existing pane
    # For subsequent values, create a new pane
    if [ $i -gt 0 ]; then
        tmux split-window -v -t $SESSION_NAME
    fi

    # Send commands separately to avoid quoting issues
    tmux send-keys -t $SESSION_NAME:0.$i "cd $DIR" C-m
    tmux send-keys -t $SESSION_NAME:0.$i "source $MYENV_PATH" C-m

    # Add a small random variation to the learning_rate to ensure the config has an effect
    tmux send-keys -t $SESSION_NAME:0.$i \
        "python train.py --config parameters.json" \
        " --custom " \
        " '{ \"log_top_dir\": \"${LOG_DIR}\", " \
        " \"options\": " \
        " {\"seed\":${values[$i]}, \"damping\": 1, " \
        " \"max_muscle_strength\": 0.5, \"muscle_increase_factor\": 1.1}, " \
        " \"env_kwargs\": {\"episode_length\": 90}, " \
        " \"hebbian_kwargs\": [{},{\"gain\": 2, \"lock_target\": 0.9}]}'" C-m

    sleep 2

    # Equalize all panes (make them the same size)
    tmux select-layout -t $SESSION_NAME even-vertical

done

    tmux select-layout -t $SESSION_NAME tiled

echo "All experiments started in tmux session: $SESSION_NAME"
