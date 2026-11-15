"""Export the TensorBoard scalars of an RL sweep for ds_process_rl.m.

Reads every event file under <root>/tb/ (eps/* and eval/* tags) and writes
<root>/df.h5, the long table (Step, Value, Tag, ModelID, Model) that
ds_process_rl loads.

Usage (repo root, environment with the analysis extra):
    python analysis/export_tb.py <root> [<root> ...]
With several roots, the combined table is written to the first one.
"""
import sys

from deep_spine.utils.analysis_tools import training_logs_to_dataframe

if len(sys.argv) < 2:
    sys.exit(__doc__)
training_logs_to_dataframe(sys.argv[1:])
