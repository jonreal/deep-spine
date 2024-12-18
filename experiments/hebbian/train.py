import numpy as np
import traceback
from scipy import signal
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
import time
import pickle
import sys
import mujoco
from pprint import pprint
import json

from deep_spine import (
    Arm26Babble
)

from deep_spine import HebbianLearner
from deep_spine.hebbian.hebbian_learner import HebbianMonitorFigure
from deep_spine.spinal_layers.ia_reflex import IaReflexSpinalLayer
import deep_spine.spinal_layers.filters.iir as iir
import deep_spine.utils.analysis_tools as tools

frame_rate = 10
log_top_dir = './logs/nominal-tuning-2/'

options = {
    'num_eps': 25,
    'eps_length': 300,
    'learning_rate': 1,
    'W0' : 0.05,
    'lock_target': None,
    'postive_only': False,
    'min_weight': 0.01,
    'max_weight': 1,
    'seed': 1,
    'damping': 0.5,
    'viscosity': 0,
    'muscle_strength_factor': 0.1,
    'muscle_increase_factor': 1.1,
    'babble_max_prob': 0.005,
    'hebbian_gain': 1,
    'k_leak': 0.1,
    'Ia_velocity_gain': 10
}

run_id = int(time.time())
log_dir = (
    f'./{log_top_dir}/{run_id}'
    f'-W0-{options['W0']}-lr-{options['learning_rate']}'
    f'-damp-{options['damping']}-'
    f'-msf-{options['muscle_strength_factor']}'
)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

sim_animation = f'./{log_dir}/sim'
sim_log = f'./{log_dir}/data/sim-data'
heb_log = f'./{log_dir}/data/heb-data'
if not os.path.exists(f'./{log_dir}/data'):
    os.makedirs(f'./{log_dir}/data')

torch.manual_seed(options['seed'])
if options['W0'] == 'identity':
    W_Ia0 = torch.eye(6)
elif options['W0'] == 'identity_noise':
    W_Ia0 = torch.clip(
        0.5*torch.eye(6) + torch.normal(mean=0.05, std=0.01, size=(6, 6)),
        0, 1
    )
else:
    W_Ia0 = torch.normal(mean=options['W0'], std=0.01, size=(6, 6))

env_kwargs = {
    #'xml_path': 'arm26.xml',
    #'xml_path': 'myo_sim/elbow/myoelbow_2dof6muscles.xml',
    'frame_rate': frame_rate,
    'save_frames_to_path': sim_animation,
    'render_in_background': False,
    'render_lossless': False,
    'task': 'babble',
    'path_to_logfile': sim_log,
    'episode_length': options['eps_length'],
    'muscle_strength_factor': options['muscle_strength_factor'],
    'babble_max_prob': options['babble_max_prob'],
    'babble_duration_range_sec': np.array([0.05, 0.1]),
    #'elevation': -15,
    #'azimuth': 0,
    #'lookat': np.array([0.35, 0, 1])
    }
spinal_kwargs =  {
    'decimation_factor':  1,
    'mechanics_filter_cutoff':  0.0,
    'II_length_gain': 1.25,
    'II_bias': -0.2,
    'Ia_II_gain': 0.2,
    'Ia_saturation_constant': 1.25,
    'Ia_velocity_gain': options['Ia_velocity_gain'],
    'muscle_bias': -0.3,
    'muscle_gain': 1 / 0.7,
    'W_Ia':  W_Ia0,
    'K': 0.3,
    'k_leak': options['k_leak']
}

hebbian_kwargs = {
    'path_to_logfile': heb_log,
    'gain': options['hebbian_gain'],
    'learning_rate': options['learning_rate'],
    'lock_target': options['lock_target'],
    'postive_only': options['postive_only'],
    'Kmean': 0.00004,
    'Klearn': 0.001,
    'hp_cut': 0.05/(200/2)
}

all_args = {
    'options': options,
    'env_kwargs': env_kwargs,
    'spinal_kwargs': spinal_kwargs,
    'hebbian_kwargs': hebbian_kwargs,
}

with open(log_dir + '/settings.pkl', 'wb') as f:
    pickle.dump(all_args, f)

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert PyTorch tensors to lists
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    else:
        return obj

# Save as JSON without reordering
all_args_serializable = convert_to_serializable(all_args)
with open(log_dir + '/settings.txt', 'w') as f:
    json.dump(all_args_serializable, f, indent=4)

#with open(log_dir + '/settings.txt', 'w') as f:
#    pprint(all_args, stream=f)

spinal_layer = IaReflexSpinalLayer(**spinal_kwargs)
env = Arm26Babble(spinal_layer=spinal_layer,**env_kwargs)
learner = HebbianLearner(
    W=spinal_layer.net.aMN.W_Ia,
    **hebbian_kwargs
)

# Figures for hebian
heb_figs = HebbianMonitorFigure(
    xs=10,
    ys=580,
    H=80,
    W=400,
    T=1,
    sim_timestep=env.model.opt.timestep,
    source=learner
)
for heb_fig in heb_figs.figs:
    env.add_fig(heb_fig)


simulation_timestep = env.model.opt.timestep
env.model.opt.gravity[:] = [0.0, 0.0, 0.0]
env.model.opt.viscosity = options['viscosity']
env.model.dof_damping[:] = options['damping']

env.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
env.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = False
env.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
env.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True

env.reset()
env.set_logfile(f'{sim_log}-{env.eps}')
learner.set_logfile(f'{heb_log}-{env.eps}')
muscle_strength_factor = options['muscle_strength_factor']

print('---\n')
print(options)
print('---\n\n')

steps_per_eps = options['eps_length'] / simulation_timestep
try:
    for _ in range(options['num_eps']):
        tstart = time.time()
        done = False
        print('---\n')
        print(
            f'Eps: {int(env.eps)}, '
            f'Muscle strength: {muscle_strength_factor:.2f}'
        )
        while not done:
            _, _, done, _, _ = env.step(action=None)
            learner.step(
                pre_synaptic=spinal_layer.act_Ia,
                post_synaptic=spinal_layer.act_aMN
            )

        tend = time.time()
        env.set_logfile(f'{sim_log}-{env.eps}')
        learner.set_logfile(f'{heb_log}-{env.eps}')
        env.reset()
        muscle_strength_factor = (
            muscle_strength_factor * options['muscle_increase_factor']
        )
        env.set_muscle_strength_factor(muscle_strength_factor)
        print(f'sim-steps per sec: {steps_per_eps / (tend - tstart):.1f}')
        print(f'sim-secs per sec: {options['eps_length'] / (tend - tstart):.1f}')
        print('---\n\n')

except Exception as e:
    tb = traceback.format_exc()
    print(f"Error during simulation: {str(e)}")
    print(f"Traceback details:\n{tb}")
finally:
    env.close()

