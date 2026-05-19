import argparse
import json
import os, sys
import time
import numpy as np
import torch
import traceback
import mujoco
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

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
    make_supraspine_viz, make_heatmap_geom, build_annotation_text, build_tb_tag
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
    "supraspine_viz": make_supraspine_viz,
}

VIZ_GEOMS = {
    "heatmap_geom": make_heatmap_geom,
}


def main():

    # Parse args
    parser = argparse.ArgumentParser(
        description="Run deep spine simulation with default and custom configs."
    )
    parser.add_argument('--config', default=None,
        help="Path to default config JSON file."
    )
    parser.add_argument('--continue', dest='continue_path', default=None,
        metavar='SIM_DIR',
        help="Continue training from a previous sim directory."
    )
    parser.add_argument('--custom', type=parse_json_or_path, default=None,
        help="Path to custom override JSON file (e.g. for options)."
    )
    parser.add_argument('--eval', nargs='?', type=int, const=100, default=None,
        metavar='NUM_EPS',
        help="Forward sim mode: run trained policy for NUM_EPS episodes (default 100).")
    parser.add_argument('--log', action='store_true', help="Enable logging.");
    parser.add_argument('--show', action='store_true',
        help="Enable rendering at 10 fps.")
    args = parser.parse_args()

    if args.continue_path is None and args.config is None:
        parser.error("one of --config or --continue is required")

    # --show injects frame_rate into custom config
    if args.show:
        show_override = {"env_kwargs": {"frame_rate": 10}}
        if args.custom is None:
            args.custom = show_override
        else:
            recursive_update(args.custom, show_override)

    # Load config: from --continue sim dir or --config file
    if args.continue_path is not None:
        sim_dir = Path(args.continue_path).resolve()
        with open(sim_dir / 'config.json') as f:
            config = json.load(f)

        # Redirect log_dir to a -continue sibling directory
        config['log_dir'] = str(sim_dir.parent) + '-continue'

        # Find latest checkpoint
        model_dir = sim_dir / 'model'
        checkpoints = sorted(
            model_dir.glob('*.zip'), key=lambda p: int(p.stem)
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {model_dir}")
        latest_ckpt = str(checkpoints[-1])
        config['supraspine_kwargs']['checkpoint'] = latest_ckpt
        config['parent_sim_id'] = sim_dir.name
        print(f'  Continuing from: {sim_dir.name}  checkpoint: {Path(latest_ckpt).name}')
    else:
        with open(args.config, 'r') as f:
            config = json.load(f)

    if args.custom:
        changed = recursive_update(config, args.custom)
        if not changed and args.continue_path is None:
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
    if args.continue_path is not None:
        all_args['parent_sim_id'] = config.get('parent_sim_id')
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(convert_to_serializable(all_args), f, indent=4)

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

    if supraspine_type in ('PPO', 'SAC'):
        supraspine_kwargs.setdefault("kwargs", {})
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
    spine.set_learners(learners)

    # Auto-generate tensorboard log dir for RL modes (after set_learners for type tag)
    if supraspine_type in ('PPO', 'SAC'):
        if args.continue_path is not None:
            # Use parent sim_id as prefix and reconstruct original custom tag
            # so continued runs group correctly with their sweep siblings
            tb_prefix = config.get('parent_sim_id', sim_id)
            parent_custom_txt = Path(args.continue_path) / 'custom.txt'
            if parent_custom_txt.exists():
                parent_custom = parse_json_or_path(str(parent_custom_txt))
                custom_for_tag = {k: v for k, v in parent_custom.items()
                                  if k not in ("log_dir", "options")}
            else:
                custom_for_tag = {}
        else:
            tb_prefix = sim_id
            custom_for_tag = {k: v for k, v in (args.custom or {}).items()
                              if k not in ("log_dir", "options")}

        custom_tag = build_tb_tag(custom_for_tag)
        tag_str = f"-{custom_tag}" if custom_tag else ""
        spine_label = f'{spine.type}_{spine.tag}' if spine.tag else spine.type
        supraspine.model.tensorboard_log = (
            f'{root_log_dir}/tb/{tb_prefix}-{spine_label}{tag_str}'
        )

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

            if args.eval is None:
                # Training mode
                mdir = f'./{log_dir}/model'
                os.makedirs(mdir, exist_ok=True)
                check_points = options.get("check_points",1)
                total_sim_steps = options.get("total_sim_steps", 1000)
                total_supraspine_steps = total_sim_steps // supraspine.decimation_factor
                steps_per_check = total_supraspine_steps // check_points
                print(f'  total_sim_steps: {total_sim_steps}  →  '
                      f'supraspine_steps: {total_supraspine_steps}')
                prev_total_steps = 0
                tb_mov_ave_eps_win = 10

                tensorboard_callback = LogCallbackArm26Pendulum(
                    env=env,
                    window=tb_mov_ave_eps_win
                )

                eval_eps  = options.get("eval_eps", 0)
                eval_seed = options.get("eval_seed", 42)

                tinit = time.time()
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

                    if eval_eps > 0:
                        print('\t--- Eval ---')
                        last_obs       = supraspine.model._last_obs.copy()
                        last_ep_starts = supraspine.model._last_episode_starts.copy()
                        ic_rng_state   = env.ic_rng.bit_generator.state
                        solved_before  = env.solved
                        eps_before     = env.eps

                        env.logging_enabled = False
                        env.ic_rng = np.random.default_rng(eval_seed)
                        obs, _ = env.reset()
                        ep_rews = []
                        for _ in range(eval_eps):
                            done = truncated = False
                            while not (done or truncated):
                                action = supraspine.predict(obs)
                                obs, reward, done, truncated, info = env.step(action)
                            ep_rews.append(env.ep_reward)
                            obs, _ = env.reset()

                        step = supraspine.model.num_timesteps
                        tensorboard_callback.logger.record("eval/mean_rew", np.mean(ep_rews))
                        tensorboard_callback.logger.record("eval/n_perfect", env.solved - solved_before)
                        tensorboard_callback.logger.dump(step)
                        print(f'  eval ({eval_eps} eps): mean_rew={np.mean(ep_rews):.1f}  '
                              f'n_perfect={env.solved - solved_before}')

                        env.solved                        = solved_before
                        env.eps                           = eps_before
                        env.ic_rng.bit_generator.state    = ic_rng_state
                        env.logging_enabled = True
                        obs, _ = env.reset()
                        supraspine.model._last_obs        = last_obs
                        supraspine.model._last_episode_starts = last_ep_starts

                    print('---\n\n')

                print('\n---\n')
                print(f'Total sim time: {(tend - tinit):.1f} (sec), '
                      f'({(tend - tinit) / 60:.1f} min)')
                print('\n---\n')

            else:
                # Forward sim — deterministic policy evaluation
                num_eps = args.eval
                dt = env.model.opt.timestep
                tb_dir = supraspine.model.tensorboard_log
                writer = SummaryWriter(log_dir=tb_dir)
                obs, _ = env.reset()
                for i in range(num_eps):
                    tstart = time.time()
                    done = False
                    while not done:
                        action = supraspine.predict(obs)
                        obs, reward, done, truncated, info = env.step(action)
                    sim_steps = env.data.time / dt
                    obs, _ = env.reset()
                    writer.add_scalar("eps/ep_rew", env.log_eps_reward, i)
                    writer.add_scalar("eps/solved", env.solved, i)
                    writer.add_scalar("eps/ic_q0", env.log_ic_q0, i)
                    writer.add_scalar("eps/ic_q1", env.log_ic_q1, i)
                    writer.add_scalar("eps/fc_q0", env.log_fc_q0, i)
                    writer.add_scalar("eps/fc_q1", env.log_fc_q1, i)
                    writer.add_scalar("eps/fc_pend", env.log_fc_pend, i)
                    tend = time.time()
                    print(f'Eps {i+1}/{num_eps}: '
                          f'sim-steps/sec: {sim_steps/(tend-tstart):.1f}')
                writer.close()

        else:

            steps_per_eps = env.episode_length / env.model.opt.timestep
            last_write_time = 0.0
            tinit = time.time()
            data_file = open(f'./{log_dir}/data.txt', 'wb')

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

                    current_time = int(
                        env.data.time + 1/env.simulation_sampling_rate
                    )

                    if (
                        current_time > 0
                        and current_time > int(last_write_time)
                    ):
                        all_W = (
                            torch.cat([l.W.flatten() for l in learners])
                        )
                        row = np.concatenate([
                            np.array([i+1, current_time, muscle_strength],
                                     dtype=np.float64),
                            all_W.detach().cpu().numpy().astype(np.float64)
                        ])
                        data_file.write(row.tobytes())
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
                print('---\n\n')

            print('\n---\n')
            print(f'Total sim time: {(tend - tinit):.1f} (sec), '
                  f'({(tend - tinit) / 60:.1f} min)')
            print('\n---\n')
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error during simulation: {str(e)}")
        print(f"Traceback details:\n{tb}")
    finally:
        if 'data_file' in dir():
            data_file.close()
        env.close()

if __name__ == '__main__':
    main()
