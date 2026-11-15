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

## Canonical run: per-step signals and Granger analysis

Needs a run with `--log` and its rule-computed weight update:

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

## Paper figures (`pubfigs/`)

Each function reads `<repo>/logs` and writes to `pubfigs/out/` by default
(`figN(data_root, out_dir)`). Sweep figures read the summary tables, so run
`ds_process_babble` on each sweep folder first. To make everything, loading the
canonical run once:

```matlab
addpath('<repo>/analysis/pubfigs')
make_all
```

| Figure | Function | Data | Output |
|---|---|---|---|
| Fig 1 | — (drawn by hand) | | |
| Fig 2 | `fig2_motif` | `hebbian-sweep/SR0-MR0{,-gain}` | `motif-regression.pdf` |
| Fig 3 | `fig3_grid` | canonical run | `grid.pdf` |
| Fig 4 | `fig4_granger` | canonical run + `dw_rule.bin` | `granger.pdf` |
| Fig 5 | `fig5_sweep` | `hebbian-sweep/SR*-MR0` | `sweep.pdf` |
| Figs S1, S2 | `figS1S2_kappa` | `hebbian-sweep/SR0-MR0{,-gain,-gain-10}` | `si-kappa-lines.pdf`, `si-kappa-heatmaps.pdf` |
| Fig S3 | `figS3_timehistory` | canonical run | `si-timehistory.pdf` |
| Fig S4 | `figS4_sweep_mr1` | `hebbian-sweep/SR*-MR1` | `si-sweep-mr1.pdf` |
| Fig S5 | `figS5_plant` | `out/si_plant.mat` from `python analysis/pubfigs/export_plant.py` | `si-plant.pdf` |
| Fig S6 | `figS6_ia_lag` | canonical run + `dw_rule.bin` | `si-ia-lag.pdf` |
| Table S1 | `tableS1_motif` | `hebbian-sweep/SR*-MR*` | `si-motif-table.tex` |
| Movies S1–S4 | `cut_movies.py` | canonical `sim.mp4` | `si-movie-{start,early,late,end}.mp4` |

`fig4_granger` and `figS6_ia_lag` print the statistics reported in the text.
