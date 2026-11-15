# deep-spine

Neuromechanical simulation of spinal circuits and their self-organization.
deep-spine couples a MuJoCo musculoskeletal model to a spinal network of
afferents, interneurons and motoneurons, driven by a supraspinal controller,
with optional Hebbian plasticity on the spinal connections. It can be used to
study how reflex circuits self-organize during motor babble, or to train a
controller on top of a (learned or fixed) spine with reinforcement learning.

> **To reproduce the results of the accompanying paper, see
> [REPRODUCING.md](REPRODUCING.md).**

- **Plant:** designed for MuJoCo models with muscle actuators: the spine sizes
  itself from the model's muscles and reads their lengths, velocities and forces
  in MuJoCo's own normalization. Tested with the included `arm26`, a planar
  two-joint arm with six muscles (shoulder, elbow and biarticular
  flexor/extensor pairs), and a variant carrying an inverted pendulum.
- **Spine:** `ia_reflex` (Ia afferents → motoneurons), `ia_interneurons`
  (adds excitatory and inhibitory interneuron populations), `identity` (no
  spinal processing).
- **Supraspine:** `babble` (random motor babble), `SAC` (Stable-Baselines3,
  with sensory and motor delays) and `PPO`.
- **Plasticity:** Hebbian learners on any weight matrix in the network.
- **Output:** binary logs for MATLAB/Python analysis, videos with live network
  and weight overlays.

## Install

Python 3.13. Dependencies are pinned (`pyproject.toml`, `uv.lock`).

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

The launch scripts also need [GNU parallel](https://www.gnu.org/software/parallel/)
(`brew install parallel` or `apt install parallel`).

## Running

From the repository root:

```bash
deep-spine --config configs/babble.json          # 2000 episodes of motor babble
deep-spine --config configs/babble.json --show   # same, rendered to video
deep-spine --config configs/babble.json --custom '{"options": {"num_eps": 10}}'
```

| Option | Effect |
|---|---|
| `--config FILE` | JSON configuration (see below) |
| `--custom JSON` | JSON string or file merged over the config; its values are also printed on the video |
| `--log` | also write per-step signals and weights (large: ~21 GB for 2000 × 90 s episodes) |
| `--show` | render to video at 10 frames per simulated second |
| `--eval [N]` | run a trained RL policy deterministically for N episodes |
| `--continue DIR` | continue training from a previous run |

Each run writes to `<log_dir>/<sim-id>/` (sim ID = start time in seconds):

| File | Contents |
|---|---|
| `config.json`, `custom.txt` | full configuration and the `--custom` overrides |
| `data.txt` | 1 Hz log: episode, time, muscle strength, Hebbian weights (float64) |
| `env.txt`, `hebbian.txt` | per-step signals and weights (with `--log`) |
| `sim.mp4` | video (with `--show` or a `frame_rate` in the config) |
| `runtime.log` | console output |

Quick installation check (10 minutes on 23 cores; 200 short runs):

```bash
./launch/test-sweep.sh       # -> logs/test-sweep/
```

## Configuration

A config is one JSON file with these sections (see `configs/babble.json`):

| Section | Contents |
|---|---|
| `log_dir` | output folder (relative to the repository root) |
| `options` | episodes, seed, muscle-strength schedule (`max_muscle_strength`, `strength_shape`, `ramp_fraction`) |
| `env_kwargs` | environment `type`, model (`xml_path`), viscosity, damping, gravity, body masses, relative muscle strengths, episode length, video settings |
| `spine_kwargs` | spine `type` and its gains and filters |
| `supraspine_kwargs` | controller `type` (babble, SAC, PPO) and its parameters |
| `hebbian_kwargs` | list of learners: which weights (`"spine.net.W_Ia"`), pre/post signals, gain, learning rate, filters |
| `viz` | figures and 3-D overlays drawn on the video |

Strings such as `"spine.net.W_Ia"` resolve to objects in the running
simulation, so learners and visualizations can target any part of the network.

Included configs: `babble.json` (Hebbian babble in the six-muscle arm) and
`babble-mutant.json` (same, with altered relative muscle strengths).

To try another model, set `env_kwargs.xml_path` to its MJCF file. Expect some
work: spinal gains are tuned for `arm26`'s muscle operating range, the
environments contain `arm26`-specific settings (e.g. body-mass overrides), and
the analysis and visualization tools assume six muscles by default.

## Analysis

MATLAB tools for sweeps, per-step signals and the Granger analysis, plus
Python export scripts: see [analysis/README.md](analysis/README.md).

## Reproducing the paper

Simulations, analysis and figures for the accompanying paper:
[REPRODUCING.md](REPRODUCING.md).

## Layout

```
deep_spine/      simulation package (environments, spine, supraspine, hebbian, utils)
configs/         example configurations
launch/          sweep launchers
analysis/        analysis library and paper figure scripts
logs/            simulation output (not tracked)
```

## License

Pending (UC Riverside Office of Technology Partnerships).

## Citation

If you use deep-spine, please cite (see also [CITATION.cff](CITATION.cff)):

> J. Realmuto, E. Nozari, G. E. Loeb. Hebbian self-organization of monosynaptic
> reflexes in a biologically realistic multiarticular limb.
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) (2026).
