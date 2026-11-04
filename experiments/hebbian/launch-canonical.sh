#!/bin/bash

# Canonical run with video, matching the SI time traces (Fig. S2).
#
# p-canonical.json reproduces logs/example-3/1790298957 exactly (SR0-MR0,
# kappa=5, lambda_max=0.2, p_b=0.0046, 2000 x 90 s episodes, seed 1) and
# renders at frame_rate=20, CRF 23, with the white->red weight map. The frame
# rate is set in the config, not via --show, and no --custom is passed, so the
# video carries no parameter annotation.
#
# Output: logs/canonical/<sim-id>/sim.mp4 (full 50 h run, ~25 GB; written at
# 10 x frame_rate = 200 fps, i.e. 10x real time). Cut the SI clips afterwards
# with analysis/si_cut_movies.py (Fig. S2 windows; video time = sim time / 10).
#
# Usage (from experiments/hebbian):
#   ./launch-canonical.sh
# Add --log to also write the 200 Hz env.txt / hebbian.txt (~20 GB).

cd "$(dirname "$0")"
source ../../myenv/bin/activate
python run.py --config p-canonical.json "$@"
