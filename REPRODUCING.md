# Reproducing the paper

Simulations, analysis and figures for:

> J. Realmuto, E. Nozari, G. E. Loeb. Hebbian self-organization of monosynaptic
> reflexes in a biologically realistic multiarticular limb.
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) (2026).

Install as in the [README](README.md); the pinned versions are the ones that
produced the published results, and the simulations are seeded and
deterministic. Run everything from the repository root.

## 1. Simulations

| Launcher | Runs | Output | Time (23 jobs) | Disk |
|---|---|---|---|---|
| `launch/sweep-embodiment.sh` | 600: 6 bodies × 10 strengths × 10 babble levels, κ = 5 | `logs/hebbian-sweep/SR{0,1,2}-MR{0,1}/` | ~3 days | ~32 GB |
| `launch/sweep-kappa.sh` | 500: SR0-MR0, κ = 1–4 and 10 | `logs/hebbian-sweep/SR0-MR0-gain{,-10}/` | ~2.5 days | ~27 GB |
| `launch/canonical.sh` | 1: the canonical run with per-step logs and video, then `dw_rule.bin` and Movies S1–S4 | `logs/hebbian-canonical/` | ~18 h | ~46 GB |

Every run is `configs/babble.json` with the sweep's overrides: 2000 episodes of
90 s (50 h simulated), about 2.6 h each. The canonical run is the SR0-MR0 sweep
point at strength 0.2, babble 0.0046.

Bodies: SR = relative muscle strength, shoulder : elbow : biarticular (SR0 =
1 : 0.5 : 0.25, SR1 = 1 : 0.25 : 0.5, SR2 = 0.5 : 0.5 : 0.75); MR = proximal :
distal mass (MR0 = 2 : 1, MR1 = 1.5 : 1.5).

Options: a body list (`./launch/sweep-embodiment.sh SR2-MR0`), `LOG_ROOT=...` to
write elsewhere, `JOBS=N` (default: one per core, minus one).

**Never press Ctrl-C under `parallel`**: it sends SIGTERM, the video is not
finalized and the mp4 is unreadable.

## 2. Analysis and figures (MATLAB)

Requirements and the analysis library: [analysis/README.md](analysis/README.md).

```matlab
addpath('<repo>/analysis', '<repo>/analysis/pubfigs')

% summary tables for every sweep folder
for c = {'SR0-MR0','SR1-MR0','SR2-MR0','SR0-MR1','SR1-MR1','SR2-MR1','SR0-MR0-gain','SR0-MR0-gain-10'}
  ds_process_babble(fullfile('<repo>/logs/hebbian-sweep', c{1}));
end

make_all          % every figure and Table S1 -> analysis/pubfigs/out/
```

Fig S5 needs the plant data first: `python analysis/pubfigs/export_plant.py`.
Each figure can also be made on its own, e.g. `fig4_granger()`.

| Figure | Function | Data | Output |
|---|---|---|---|
| Fig 1 | drawn by hand | | |
| Fig 2 | `fig2_motif` | `hebbian-sweep/SR0-MR0{,-gain}` | `motif-regression.pdf` |
| Fig 3 | `fig3_grid` | canonical run | `grid.pdf` |
| Fig 4 | `fig4_granger` | canonical run + `dw_rule.bin` | `granger.pdf` |
| Fig 5 | `fig5_sweep` | `hebbian-sweep/SR*-MR0` | `sweep.pdf` |
| Figs S1, S2 | `figS1S2_kappa` | `hebbian-sweep/SR0-MR0{,-gain,-gain-10}` | `si-kappa-lines.pdf`, `si-kappa-heatmaps.pdf` |
| Fig S3 | `figS3_timehistory` | canonical run | `si-timehistory.pdf` |
| Fig S4 | `figS4_sweep_mr1` | `hebbian-sweep/SR*-MR1` | `si-sweep-mr1.pdf` |
| Fig S5 | `figS5_plant` | `si_plant.mat` from `export_plant.py` | `si-plant.pdf` |
| Fig S6 | `figS6_ia_lag` | canonical run + `dw_rule.bin` | `si-ia-lag.pdf` |
| Table S1 | `tableS1_motif` | `hebbian-sweep/SR*-MR*` | `si-motif-table.tex` |
| Movies S1–S4 | `cut_movies.py` (run by `canonical.sh`) | canonical `sim.mp4` | `si-movie-{start,early,late,end}.mp4` |

`fig4_granger` and `figS6_ia_lag` print the statistics reported in the text.
Cutting the movies needs `ffmpeg`.
