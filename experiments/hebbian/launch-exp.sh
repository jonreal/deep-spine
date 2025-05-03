#!/bin/bash

# Array of values for lock_target
values=(0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

# Directory path
DIR="/Users/jonreal/research/deep-spine/experiments/hebbian"
MYENV_PATH="../../myenv/bin/activate"

# Kill any existing tmux session with the same name
tmux kill-session -t experiments 2>/dev/null

# Start a new tmux session
tmux new-session -d -s experiments

# For each value, create a pane and run the command
for ((i=0; i<${#values[@]}; i++)); do
    # For the first value, use the existing pane
    # For subsequent values, create a new pane
    if [ $i -gt 0 ]; then
        tmux split-window -h -t experiments
        tmux select-layout -t experiments tiled
    fi
    
    # Send commands separately to avoid quoting issues
    tmux send-keys -t experiments:0.$i "cd $DIR" C-m
    tmux send-keys -t experiments:0.$i "source $MYENV_PATH" C-m
    tmux send-keys -t experiments:0.$i "python train.py --config parameters.json --custom '{\"hebbian_kwargs\":[{},{\"lock_target\":${values[$i]}}]}'" C-m
done

# Attach to the session
tmux attach-session -t experiments
