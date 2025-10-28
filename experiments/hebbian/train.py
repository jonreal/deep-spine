import argparse
import json
import os, shutil, sys
import time
import pickle
import numpy as np
import torch
import traceback
import mujoco
from scipy import signal
from pprint import pprint
from pathlib import Path
from scipy import signal

# Deep spine imports
from deep_spine import Arm26Babble, HebbianLearner
from deep_spine.spine.ia_reflex import IaReflexSpine
from deep_spine.spine.filters.fir import FirFilt
from deep_spine.spine.filters.iir import IirFilt
from deep_spine.supraspine.babble import Babble
from deep_spine.hebbian.hebbian_learner import HebbianMonitorFigure
import deep_spine.utils.viz as viz
from deep_spine.utils.viz import make_spine_viz, make_hebbian_viz


def recursive_update(d, u, *, strict=False):
    changed = False

    # Special handling for lists of dictionaries (like hebbian_kwargs)
    if isinstance(d, list) and isinstance(u, list):
        # Make sure the lists are of the same length
        if len(d) != len(u):
            raise ValueError(f"[config error] Cannot update a list of length {len(d)} with a list of length {len(u)}")

        # Update each dictionary in the list
        for i in range(len(d)):
            if isinstance(d[i], dict) and isinstance(u[i], dict):
                changed |= recursive_update(d[i], u[i], strict=strict)
            else:
                if d[i] != u[i]:
                    d[i] = u[i]
                    changed = True
        return changed

    # Original code for dictionaries
    if isinstance(d, dict) and isinstance(u, dict):
        for k, v in u.items():
            if strict and k not in d:
                raise KeyError(f"[config error] unknown key '{k}' in override")
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                changed |= recursive_update(d[k], v, strict=strict)
            elif isinstance(v, list) and k in d and isinstance(d[k], list):
                # Handle nested lists (like in hebbian_kwargs)
                changed |= recursive_update(d[k], v, strict=strict)
            else:
                if k not in d or d[k] != v:
                    d[k] = v
                    changed = True
        return changed

    # Default case for non-dictionary, non-list values
    if d != u:
        d = u
        changed = True

    return changed

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    else:
        return obj

def format_value(v):
    if isinstance(v, (int, float)):
        return f'{v:.3g}'
    elif isinstance(v, str):
        return v[:6]
    elif isinstance(v, list):
        return "x".join(format_value(x) for x in v)
    else:
        return str(v)

def build_annotation_text(config, prefix=""):
    if config is None:
        return ""
    lines = []
    # Handle dictionaries
    if isinstance(config, dict):
        for k, v in sorted(config.items()):
            k_prefix = f"{prefix}_{k[:3]}" if prefix else k[:3]
            if isinstance(v, dict):
                # Recursive call for nested dictionaries
                nested_lines = build_annotation_text(v, k_prefix)
                lines.extend(nested_lines.split("\n"))
            elif isinstance(v, list):
                # Handle list values
                for i, item in enumerate(v):
                    if isinstance(item, (dict, list)):
                        # Use index in the prefix for nested structures
                        item_prefix = f"{k_prefix}_{i}"
                        nested_lines = build_annotation_text(item, item_prefix)
                        lines.extend(nested_lines.split("\n"))
                    else:
                        # Simple list items
                        lines.append(f"{k_prefix}[{i}]={format_value(item)}")
            else:
                # Simple key-value pairs
                lines.append(f"{k_prefix}={format_value(v)}")
    # Handle lists
    elif isinstance(config, list):
        for i, item in enumerate(config):
            if isinstance(item, (dict, list)):
                # Use index in the prefix for nested structures
                item_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                nested_lines = build_annotation_text(item, item_prefix)
                lines.extend(nested_lines.split("\n"))
            else:
                # Simple list items
                index_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                lines.append(f"{index_prefix}={format_value(item)}")
    # Handle simple types
    else:
        lines.append(f"{prefix}={format_value(config)}")
    return "\n".join(line for line in lines if line)

def initialize_weight_matrix(param_value, param_name, default_value=0.2):
    """Helper function to initialize weight matrices for both Wp0 and Wn0"""
    if param_value == 0:  # New case: If parameter is 0, return zero matrix
        return torch.zeros(6, 6, dtype=torch.float32)
    elif isinstance(param_value, str) and os.path.isfile(param_value):
        # Load from file
        data = np.loadtxt(param_value, delimiter=',')
        last_row = data[-1]
        return torch.tensor(last_row[-36:].reshape(6, 6), dtype=torch.float32)
    else:
        # Generate random values
        return torch.normal(
            mean=options.get(param_name, default_value),
            std=0.01,
            size=(6, 6)
        )

def make_weight_matrix(options, key, default_value=0.2):
    value = options.get(key)

    # Zero case
    if value == 0:
        return torch.zeros(6, 6, dtype=torch.float32)

    # File path case
    if isinstance(value, str) and os.path.isfile(value):
        data = np.loadtxt(value, delimiter=',')
        last_row = data[-1]
        return torch.tensor(last_row[-36:].reshape(6, 6), dtype=torch.float32)

    # Default/random case
    mean_value = default_value if value is None else value
    return torch.normal(mean=mean_value, std=0.01, size=(6, 6))

def parse_json_or_path(s):
    if os.path.isfile(s):
        with open(s) as f:
            return json.load(f)
    return json.loads(s)

def main():
    parser = argparse.ArgumentParser(
        description="Run deep spine simulation with default and custom configs."
    )
    parser.add_argument('--config', required=True,
                        help="Path to default config JSON file.")
    parser.add_argument('--custom', type=parse_json_or_path, default=None,
                        help="Path to custom override JSON file (e.g. for options).")
    parser.add_argument('--log', action='store_true', help="Enable logging.");
    args = parser.parse_args()

    # Load the default configuration
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

    # Unpack configuration
    options = config.get("options", {})
    env_kwargs = config.get("env_kwargs", {})
    spine_kwargs = config.get("spine_kwargs", {})
    supraspine_kwargs = config.get("supraspine_kwargs", {})
    hebbian_kwargs = config.get("hebbian_kwargs", {})
    log_top_dir = config.get("log_top_dir", "./logs/untitled")

    if args.log:
        env_log = os.path.join(log_dir, "env.txt")
        hebbian_log = os.path.join(log_dir, "hebbian.txt")
        env_kwargs["path_to_log"] = env_log
        hebbian_kwargs[1]["path_to_log"] = hebbian_log

    date_string = f'{time.strftime("%b-%-d-%Y", local_time)}\n{sim_id}'
    sim_animation = f'./{log_dir}/sim'

    torch.manual_seed(options.get("seed", 1))
    min_weight = hebbian_kwargs[1]["min_weight"];
    max_weight = hebbian_kwargs[1]["max_weight"];

    Wp0 = make_weight_matrix(options, "Wp0")
    Wn0 = make_weight_matrix(options, "Wn0")

    env_kwargs.update({
        'save_frames_to_path': sim_animation,
    })

    spine_kwargs.update({'Wp_Ia': Wp0, 'Wn_Ia': Wn0})

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

    # Initialize simulation components
    spine = IaReflexSpine(**spine_kwargs)
    supraspine = Babble(**supraspine_kwargs)
    env = Arm26Babble(spine=spine, supraspine=supraspine, **env_kwargs)

    learners = [
        HebbianLearner(
            W=eval(kwargs["W"]),
            pre=eval(kwargs["pre"]),
            post=eval(kwargs["post"]),
            **{k: v for k, v in kwargs.items() if k not in {"W", "pre", "post"}}
        )
        for kwargs in hebbian_kwargs
    ]

    # Set up visualizations
    for fig in make_spine_viz(
            data=env.data, spine=spine, supraspine=supraspine,
            xs=900, ys=900, T=1, H=80, W=300,
            timestep=env.model.opt.timestep):
        env.add_fig(fig)
    #for fig in make_hebbian_viz(
    #        learner=learners[1],
    #        xs=200, ys=540, T=1, H=65, W=180,
    #        timestep=env.model.opt.timestep, legend=True, showlabel=False):
    #    env.add_fig(fig)
    for fig in make_hebbian_viz(
            learner=learners[1],
            xs=20, ys=540, T=1, H=65, W=180,
            timestep=env.model.opt.timestep, legend=False):
        env.add_fig(fig)

    # Weight matrix visualizations
    #W_Ia_heatmap = viz.HeatmapGeom(
    #    center= env_kwargs['lookat'] + np.array([-0.9, -0.5, -0.20]),
    #    size=0.05,
    #    min_weight=0,
    #    max_weight=max_weight,
    #    W=spine.net.Wp_Ia,
    #    title='excitatory',
    #)
    #env.add_geom(W_Ia_heatmap)
    #W0 = spine.net.Wp_Ia.clone().detach()
    #diffW_Ia_heatmap = viz.HeatmapGeom(
    #    center= env_kwargs['lookat'] + np.array([-0.6, -0.1, -5]),
    #    size=0.03,
    #    W=spine.net.Wp_Ia,
    #    W0=W0,
    #    min_weight=0,
    #    max_weight=max_weight,
    #    notext=True,
    #    spacing=2
    #)
    #env.add_geom(diffW_Ia_heatmap)

    W_Ia_heatmap = viz.HeatmapGeom(
        center= env_kwargs['lookat'] + np.array([-0.9, -1.25, -0.20]),
        size=0.05,
        min_weight=0,
        max_weight=1,
        W=spine.net.Wp_Ia,
        xlabel=['Ia_SF', 'Ia_SE', 'Ia_EF', 'Ia_EE', 'Ia_BF', 'Ia_BE'],
        ylabel=['MN_SF', 'MN_SE', 'MN_EF', 'MN_EE', 'MN_BF', 'MN_BE'],
        title='excitatory'
    )
    env.add_geom(W_Ia_heatmap)
    W0 = spine.net.Wn_Ia.clone().detach()
    diffW_Ia_heatmap = viz.HeatmapGeom(
        center= env_kwargs['lookat'] + np.array([-0.6, -2.1, -5]),
        size=0.03,
        W=spine.net.Wp_Ia,
        W0=W0,
        min_weight=0,
        max_weight=1,
        notext=True,
        spacing=2
    )
    env.add_geom(diffW_Ia_heatmap)

    fig = viz.BarsFigure(
        viewport=mujoco.MjrRect(250, 0, 150, 100),
        nbars=1,
        title='Muscle Strength',
        source=env.muscle_strength,
        ylim=np.array([-1, 1]),
        xlim=np.array([0, 1]),
        gridsize=np.array([5, 2]),
        ysymmetric=True
    )
    fig.figure.flg_selection = 1
    fig.figure.selection = env.muscle_strength.item()
    env.add_fig(fig)

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

    # Simulation settings
    env.model.opt.gravity[:] = [0.0, 0.0, 0.0]
    env.model.opt.viscosity = options.get("viscosity", 0)
    env.model.dof_damping[:] = options.get("damping", 1)
    env.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
    env.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = False
    env.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    env.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True

    muscle_strength = env.muscle_strength[0]
    muscle_increase_factor = options.get("muscle_increase_factor", 1.1)
    max_muscle_strength = options.get("max_muscle_strength", 1.0)

    env.reset()
    steps_per_eps = env.episode_length / env.model.opt.timestep
    last_write_time = 0.0
    tinit = time.time()
    print(date_string);

    N1 = env.episode_length/10;
    movavg1 = FirFilt(np.ones(int(N1))/N1)

    N2 = env.episode_length;
    movavg2 = FirFilt(np.ones(int(N2))/N2)

    fc = 0.01
    b, a = signal.butter(1, fc/(1/2),'low')
    lowpass = IirFilt(b, a)



    try:
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


                    wp = spine.net.Wp_Ia.sum() / spine.net.Wp_Ia.numel()
                    avg = movavg1.filter(wp)
                    ddt_avg = 1/(2*np.pi*fc)*(avg - lowpass.filter(avg))
                    criteria = movavg2.filter(ddt_avg)
                    #print(wp, avg, ddt_avg, convergence)

                    _Wn_Ia = ",".join(
                        map(str, spine.net.Wn_Ia.flatten().tolist())
                    )
                    _Wp_Ia = ",".join(
                        map(str, spine.net.Wp_Ia.flatten().tolist())
                    )
                    _muscle_strength = muscle_strength
                    with open(f"./{log_dir}/data.txt", "a") as f:
                        f.write(
                            f"{i+1},"
                            f"{current_time:.2f},"
                            f"{_muscle_strength},"
                            f"{_Wn_Ia},"
                            f"{_Wp_Ia}\n"
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
            print(f'convergence criteria (epsilon): {abs(criteria):.6f}'
                  f' ({options.get("epsilon")})')
            print('---\n\n')

            if abs(criteria) < options.get("epsilon"):
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
