# Analysis

MATLAB analysis of deep-spine runs, and the scripts that make the paper's
figures. Tested with MATLAB R2024b and the Signal Processing and Statistics and
Machine Learning Toolboxes. Add this folder to the path:

```matlab
addpath('<repo>/analysis')
```

Library functions are prefixed `ds_` (deep-spine output) or `gc_` (Granger
analysis).

## Babble sweeps: dashboards and summary tables

Run on every sweep condition folder (one folder of run directories):

```matlab
root = '<repo>/logs/hebbian-sweep/SR0-MR0';
T = ds_process_babble(root, printfig=true, FigVis='off');
ds_plot_motif_heatmaps(T);           % R, S, A and S-A vs (babble, strength), one row per kappa
ds_plot_motif_gui(T);                % same, click a point to open that run's dashboard
```

```
ds_process_babble(root)
  ├─ ds_read_params → ds_flatten_struct   custom.txt → sweep parameters
  ├─ ds_read_data                         data.txt → per-second weights, strength
  ├─ ds_fit_motif                         motif regression W ≈ B·c (reflex r, synergy s, antagonist a)
  └─ ds_plot_dashboard                    one dashboard per run (printfig=true)
→ <root>/analysis/summary/summary_table.mat
→ <root>/analysis/dashboard/<sim-id>_<params>.png
```

`ds_find_run(root, gain=5, strength=0.2, babble=0.0046)` returns the run
directory for a parameter set.

## Per-step signals and Granger analysis

Needs a run made with `--log`, plus its rule-computed weight update (example: the canonical run):

```bash
python analysis/export_dw_rule.py logs/hebbian-canonical/<sim-id>   # writes dw_rule.bin
```

```matlab
root = '<repo>/logs/hebbian-canonical/1790298957';
[S, W, P] = ds_load_trial(root);     % 200 Hz signals, flat weights, metadata (~21 GB)
dW = ds_load_dw_rule(root);          % per-step update before float32 rounding
R  = gc_batch(S, W, loFrac=0, hiFrac=0.9, lag=0.4, dWrule=dW);
```

Weights are float32 in the simulation, so differencing `W` gives a quantized
staircase; pass `dWrule=dW` to every `gc_` function.

| Function | Purpose |
|---|---|
| `gc_batch` | event-triggered Granger test (kinematics beyond Ia + history) for every connection |
| `gc_run` | the nested regression for one connection |
| `gc_weight_events` | weight-increase events used by the test |
| `gc_event_grid` | 6×6 weight traces with the selected events (gating check) |
| `gc_lag_estimation` | Ia → dw peak lag per connection (Fig S6) |
| `gc_lag_sweep` | script: test statistics vs history window length |
| `gc_dw_autocorr` | autocorrelation of dw, the reason for the history term |
| `gc_event_triggered_plot`, `gc_event_triggered_stack`, `gc_histogram` | event-triggered traces and event-size distributions |
| `ds_w_trace`, `ds_w_size` | one connection's weight trace; weight dimensions |

## RL runs

```bash
python analysis/export_tb.py <sweep-root>       # TensorBoard scalars → <root>/df.h5
```

```matlab
ds_process_rl('<sweep-root>');
```

`export_tb.py` needs the analysis extra (`uv sync --extra analysis`, or
`pip install -e ".[analysis]"`). `ds_read_pend_data` reads the 200 Hz `env.txt`
of a pendulum run.

## Other

`FLV(lmin, lmax, vmax, fpmax, fvmax)` plots MuJoCo's muscle force-length-velocity
surface (DeepMind, Apache 2.0; see the file header).

## Paper figures (`pubfigs/`)

One function per figure (`figN(data_root, out_dir)`, defaults `<repo>/logs` and
`pubfigs/out/`); `make_all` makes them all. Steps and the figure table:
[REPRODUCING.md](../REPRODUCING.md).
