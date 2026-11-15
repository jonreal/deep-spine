# deep-spine

Neuromechanical simulation of spinal circuits and their self-organization. A
two-joint, six-muscle arm (MuJoCo) is driven by random motor babble through a
spinal network of Ia afferents and motoneurons; Hebbian plasticity shapes the
Ia→MN connectivity during development.

This repository contains the code for:

> J. Realmuto, E. Nozari, G. E. Loeb. Hebbian self-organization of monosynaptic
> reflexes in a biologically realistic multiarticular limb. 2026 (in review).

## Install

Python 3.13. Dependencies are pinned to the versions that produced the
published results; simulations are seeded and deterministic.

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jonreal/deep-spine.git
cd deep-spine
uv sync                       # creates .venv with the locked versions
source .venv/bin/activate     # or prefix commands with `uv run`
```

With pip:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The sweep launchers also need [GNU parallel](https://www.gnu.org/software/parallel/)
(`brew install parallel` or `apt install parallel`), and cutting the SI movies
needs `ffmpeg`. The analysis is in MATLAB; see [analysis/README.md](analysis/README.md).

## Quick start

Run from the repository root:

```bash
deep-spine --config configs/canonical.json         # canonical run
deep-spine --config configs/mutant.json            # same, mutant strength ratio
deep-spine --config configs/canonical.json --show  # render to video at 10 fps
```

Each run writes to `logs/<name>/<sim-id>/`, where the sim ID is the start time
in seconds:

| File | Contents |
|---|---|
| `config.json`, `custom.txt` | full configuration and the `--custom` overrides |
| `data.txt` | 1 Hz log: episode, time, muscle strength, Ia→MN weights (float64) |
| `env.txt`, `hebbian.txt` | 200 Hz signals and weights; only with `--log` (~21 GB per run) |
| `sim.mp4` | video; only when `frame_rate` is set (or `--show`) |
| `runtime.log` | console output |

`--custom` takes a JSON string (or file) merged over the config, e.g.
`--custom '{"options": {"num_eps": 10}}'`. Its values are also stamped on the
video.

To check an installation end to end (about 10 min on 23 cores), run a
10-episode version of one sweep and process it as in
[analysis/README.md](analysis/README.md):

```bash
./launch/test-sweep.sh                 # -> logs/test-sweep/SR0-MR0/
```

## Reproducing the paper

| Launcher | Runs | Output | Time (23 jobs) | Disk |
|---|---|---|---|---|
| `launch/sweep-embodiment.sh` | 600: 6 bodies × 10 strengths × 10 babble levels, κ = 5 | `logs/hebbian-sweep/SR{0,1,2}-MR{0,1}/` | ~3 days | ~32 GB |
| `launch/sweep-kappa.sh` | 500: SR0-MR0, κ = 1–4 and 10 | `logs/hebbian-sweep/SR0-MR0-gain{,-10}/` | ~2.5 days | ~27 GB |
| `launch/canonical.sh` | 1: canonical run with 200 Hz logs and video, then `dw_rule.bin` and Movies S1–S4 | `logs/hebbian-canonical/` | ~18 h | ~46 GB |

Each run is 2000 episodes of 90 s (50 h simulated); a sweep run takes about
2.6 h. Launchers take an optional body list
(`./launch/sweep-embodiment.sh SR2-MR0`), `LOG_ROOT=...` to write elsewhere and
`JOBS=N` (default: one per core, minus one). The canonical run is the SR0-MR0
sweep point at strength 0.2, babble 0.0046.

**Never press Ctrl-C under `parallel`**: it sends SIGTERM, the video is not
finalized, and the mp4 is unreadable. Stop video runs from a foreground
`deep-spine` command instead.

Figures and Table S1 are then made in MATLAB from `logs/`
([analysis/README.md](analysis/README.md)).

## Layout

```
deep_spine/      simulation package (environments, spine, supraspine, hebbian)
configs/         canonical.json, mutant.json
launch/          sweep and canonical-run launchers
analysis/        MATLAB analysis library and paper figures
logs/            simulation output (not tracked)
```

## License

Pending (UC Riverside Office of Technology Partnerships).
