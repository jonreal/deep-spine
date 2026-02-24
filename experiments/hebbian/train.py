import argparse
import json
import os, shutil, sys
import time
import pickle
import numpy as np
import torch
import traceback
import mujoco
from pathlib import Path

# Deep spine imports
from deep_spine import (
    Arm26Babble, Arm26Pendulum, LogCallbackArm26Pendulum, HebbianLearner
)

from deep_spine.spine.ia_reflex import IaReflexSpine
from deep_spine.spine.ia_interneurons import IaInterneuronSpine
from deep_spine.spine.identity import IdentitySpine

from deep_spine.supraspine.babble import Babble
from deep_spine.supraspine.mlp_ppo import MlpPPO
from deep_spine.supraspine.mlp_sac import MlpSAC

import deep_spine.utils.viz as viz
from deep_spine.utils.viz import (
    make_spine_viz, make_pendulum_viz, make_hebbian_viz, make_bars_viz,
    make_heatmap_geom, build_annotation_text
)
from deep_spine.utils.config import (
    resolve_path, resolve_kwargs, recursive_update,
    convert_to_serializable, parse_json_or_path
)
from deep_spine.utils.logging import Tee
from deep_spine.utils.monitors import WindowStationarity


SPINES = {
    'ia_reflex': IaReflexSpine,
    'ia_interneurons': IaInterneuronSpine,
    'identity': IdentitySpine,
}

SUPRA = {
    'babble': Babble,
    'PPO': MlpPPO,
    'SAC': MlpSAC,
}

ENV = {
    'arm26_babble': Arm26Babble,
    'arm26_pendulum': Arm26Pendulum,
}

VIZ_FIGS = {
    "spine_viz": make_spine_viz,
    "pendulum_viz": make_pendulum_viz,
    "hebbian_viz": make_hebbian_viz,
    "bars_viz": make_bars_viz,
}

VIZ_GEOMS = {
    "heatmap_geom": make_heatmap_geom,
}


def main():

    # Parse args
    parser = argparse.ArgumentParser(
        description="Run deep spine simulation with default and custom configs."
    )
    parser.add_argument('--config', required=True,
        help="Path to default config JSON file."
    )
    parser.add_argument('--custom', type=parse_json_or_path, default=None,
        help="Path to custom override JSON file (e.g. for options)."
    )
    parser.add_argument('--log', action='store_true', help="Enable logging.");
    args = parser.parse_args()

    # Load the default configuration and custom settings
    with open(args.config, 'r') as f:
        config = json.load(f)
    if args.custom:
        changed = recursive_update(config, args.custom)
        if not changed:
            raise ValueError(
                f"[config error] custom config '{args.custom}' had no effect")

    # --- Unpack config ---
    options = config.get("options", {})
    root_log_dir = config.get("log_dir", "./logs/untitled")
    now = time.time()
    local_time = time.localtime(now)
    sim_id = int(now)

    log_dir = os.path.join(
        root_log_dir,
        f'{sim_id}'
    )
    os.makedirs(log_dir, exist_ok=True)

    logf = open(os.path.join(log_dir, "runtime.log"), "a", buffering=1)
    sys.stdout = Tee(sys.stdout, logf)
    sys.stderr = Tee(sys.stderr, logf)

    # Unpack configuration
    env_kwargs = config.get("env_kwargs", {})
    spine_kwargs = config.get("spine_kwargs", {})
    supraspine_kwargs = config.get("supraspine_kwargs", {})
    hebbian_kwargs = config.get("hebbian_kwargs", {})
    viz_config = config.get("viz", [])

    if args.log:
        env_log = os.path.join(log_dir, "env.txt")
        env_kwargs["path_to_log"] = env_log
        if hebbian_kwargs:
            hebbian_log = os.path.join(log_dir, "hebbian.txt")
            hebbian_kwargs[0]["path_to_log"] = hebbian_log

    date_string = f'{time.strftime("%b-%-d-%Y", local_time)}\n{sim_id}'
    sim_animation = f'./{log_dir}/sim'
    torch.manual_seed(options.get("seed", 1))

    print('-----------');
    print(date_string);
    print('-----------\n');

    env_kwargs.update({
        'save_frames_to_path': sim_animation,
    })

    # Save configuration for reproducibility
    all_args = {
        'options': options,
        'env_kwargs': env_kwargs,
        'spine_kwargs': spine_kwargs,
        'supraspine_kwargs': supraspine_kwargs,
        'hebbian_kwargs': hebbian_kwargs,
    }
    with open(os.path.join(log_dir, 'settings.pkl'), 'wb') as f:
        pickle.dump(all_args, f)
    with open(os.path.join(log_dir, 'settings.txt'), 'w') as f:
        json.dump(convert_to_serializable(all_args), f, indent=4)

    # Copy the --config file
    shutil.copy(args.config, os.path.join(log_dir, 'config.json'))

    # Capture the raw --custom string
    if '--custom' in sys.argv:
        idx = sys.argv.index('--custom') + 1
        custom_str = sys.argv[idx]
        custom_path = Path(custom_str)
        with open(os.path.join(log_dir, 'custom.txt'), 'w') as f:
            if custom_path.exists() and custom_path.suffix == '.json':
                with open(custom_path, 'r') as cf:
                    f.write(cf.read())
            else:
                f.write(custom_str)

    seed = options.get("seed",1)

    # Initialize simulation components
    spine_type = spine_kwargs.pop("type")
    spine = SPINES[spine_type](**spine_kwargs)

    supraspine_type = supraspine_kwargs.pop("type")

    # Auto-generate tensorboard log dir for RL modes
    if supraspine_type in ('PPO', 'SAC'):
        supraspine_kwargs.setdefault("kwargs", {})
        # Build tag from custom, excluding log_dir and options (redundant)
        custom_for_tag = {k: v for k, v in (args.custom or {}).items()
                         if k not in ("log_dir", "options")}
        custom_tag = build_annotation_text(custom_for_tag).replace("\n", "_")
        tag_str = f"-{custom_tag}" if custom_tag else ""
        supraspine_kwargs["kwargs"]["tensorboard_log"] = (
            f'{root_log_dir}/tb/{sim_id}-{spine_type}{tag_str}'
        )
        supraspine_kwargs["kwargs"]["seed"] = seed

    supraspine = SUPRA[supraspine_type](**supraspine_kwargs)
    supraspine.update_seed(seed)

    env_type = env_kwargs.pop("type")
    env = ENV[env_type](spine=spine, supraspine=supraspine, **env_kwargs)
    env.update_seed(seed)
    env.reset()

    # Hebbian Learning (optional)
    learners = []
    if hebbian_kwargs:
        resolve = lambda path: resolve_path(path, {"spine": spine})
        learners = [
            HebbianLearner(
                W=resolve(kwargs["W"]),
                pre=resolve(kwargs["pre"]),
                post=resolve(kwargs["post"]),
                **{k: v for k, v in kwargs.items() if k not in {"W", "pre", "post"}}
            )
            for kwargs in hebbian_kwargs
        ]

    # Set up visualizations from config
    viz_context = {"env": env, "spine": spine,
                   "supraspine": supraspine, "learners": learners,
                   "env_kwargs": env_kwargs}

    for item_kwargs in viz_config:
        item_kwargs = item_kwargs.copy()  # Don't mutate config
        item_type = item_kwargs.pop("type")
        resolved = resolve_kwargs(item_kwargs, viz_context)

        if item_type in VIZ_FIGS:
            for fig in VIZ_FIGS[item_type](**resolved):
                env.add_fig(fig)
        elif item_type in VIZ_GEOMS:
            for geom in VIZ_GEOMS[item_type](**resolved):
                env.add_geom(geom)

    annotation_text = build_annotation_text(args.custom)
    annotation = viz.Annotation(
        text=annotation_text,
        viewport=mujoco.MjrRect(220, 90, 200, 200),
        gridpos=mujoco.mjtGridPos.mjGRID_BOTTOM
    )
    env.add_annotation(annotation)
    annotation = viz.Annotation(
        text=date_string,
        viewport=mujoco.MjrRect(0, 0, 1200, 100),
        gridpos=mujoco.mjtGridPos.mjGRID_BOTTOM
    )
    env.add_annotation(annotation)

    # Visuals
    env.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
    env.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = False
    env.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    env.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True

    # strength increases
    muscle_strength = env.muscle_strength[0]
    muscle_increase_factor = options.get("muscle_increase_factor", 1.1)
    max_muscle_strength = options.get("max_muscle_strength", 1.0)



    try:
        if supraspine_type in ('PPO', 'SAC'):
            mdir = f'./{log_dir}/model'
            os.makedirs(mdir, exist_ok=True)
            check_points = options.get("check_points",1)
            total_timesteps = options.get("total_timesteps",1000)
            steps_per_check = int(total_timesteps / check_points)
            prev_total_steps = 0
            tb_mov_ave_eps_win = 10


            tensorboard_callback = LogCallbackArm26Pendulum(
                env=env,
                window=tb_mov_ave_eps_win
            )

            for i in range(check_points):
                tstart = time.time()
                print(f'{sim_id}: iteration {i+1}/{check_points}:')

                supraspine.model.learn(
                    total_timesteps=steps_per_check,
                    reset_num_timesteps=False,
                    progress_bar=False,
                    callback=tensorboard_callback
                )
                supraspine.model.save(
                    f'{mdir}/{supraspine.model.num_timesteps}'
                )
                tend = time.time()
                total_steps = supraspine.model.num_timesteps
                nsteps =  total_steps - prev_total_steps
                prev_total_steps = total_steps
                print(f'sim-steps per sec: {nsteps / (tend - tstart):.1f}')
                print('---\n\n')

        else:

            steps_per_eps = env.episode_length / env.model.opt.timestep
            last_write_time = 0.0
            tinit = time.time()

            convergence_monitor = WindowStationarity(
                T=int(env.episode_length), M=3,
                rtol_mu=options.get("epsilon", 1e-4),
                rtol_sig=options.get("epsilon", 1e-4),
            )

            for i in range(options.get("num_eps", 50)):
                tstart = time.time()
                done = False
                print('---\n')
                print(
                    f'Eps: {int(env.eps)}, '
                    f'Muscle strength: {muscle_strength:.2f}'
                )
                last_write_time = 0
                while not done:
                    _, _, done, _, _ = env.step(action=None)
                    for learner in learners:
                        learner.step()

                    current_time = int(
                        env.data.time + 1/env.simulation_sampling_rate
                    )

                    if current_time > 0 and current_time > int(last_write_time):

                        W = spine.net.W_Ia
                        wp = (W**2).mean().sqrt().item()
                        is_converged, info = convergence_monitor.update(wp)

                        _W_Ia = ",".join(
                            map(str, spine.net.W_Ia.flatten().tolist())
                        )
                        _muscle_strength = muscle_strength
                        with open(f"./{log_dir}/data.txt", "a") as f:
                            f.write(
                                f"{i+1},"
                                f"{current_time:.2f},"
                                f"{_muscle_strength},"
                                f"{_W_Ia}\n"
                            )
                        last_write_time = current_time
                tend = time.time()
                env.reset()
                muscle_strength = np.clip(
                    muscle_strength * muscle_increase_factor,
                    0, max_muscle_strength
                )
                env.set_muscle_strength(muscle_strength)
                print(f'sim-steps per sec: {steps_per_eps / (tend - tstart):.1f}')
                print(f'sim-secs per sec: {env.episode_length / (tend - tstart):.1f}')
                if info.get('status') == 'window_done':
                    print(f'convergence: {info["n_windows"]}/{info["M"]} windows, '
                          f'mu={info["mu"]:.4g}, sig={info["sig"]:.4g}')
                    if 'mu_span_rel' in info:
                        print(f'  mu_span={info["mu_span_rel"]:.2e} (tol={info["rtol_mu"]:.2e}), '
                              f'sig_span={info["sig_span_rel"]:.2e} (tol={info["rtol_sig"]:.2e})')
                print('---\n\n')

                if is_converged:
                    print('\n---\n')
                    print('Simulation converged')
                    print('\n---\n')
                    break;

            print('\n---\n')
            print(f'Total sim time: {(tend - tinit):.1f} (sec), '
                  f'({(tend - tinit) / 60:.1f} min)')
            print('\n---\n')
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error during simulation: {str(e)}")
        print(f"Traceback details:\n{tb}")
    finally:
        env.close()

if __name__ == '__main__':
    main()
