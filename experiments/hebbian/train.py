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

from deep_spine import (
    Arm26Babble
)

from deep_spine import HebbianLearner
from deep_spine.spinal_layers.ia_reflex import IaReflexSpinalLayer
import deep_spine.spinal_layers.filters.iir as iir
import deep_spine.utils.analysis_tools as tools

import deep_spine.utils.monitors as weight_monitor
import deep_spine.utils.scope as scope

env_scope_port =  8000
scope_port = env_scope_port + 1

frame_rate = 10
log_top_dir = './logs/arm26-11-26-24/'

options = {
    'num_eps': 10,
    'eps_length': 900,
    'save_W_period': 30,
    'learning_rate': 10,
    'W0' : 'identity_noise',
    'lock_target': None,
    'postive_only': False,
    'min_weight': 0.01,
    'max_weight': 1,
    'seed': 1,
    'damping': 0.5,
    'viscosity': 0,
    'muscle_strength_factor': 0.7,
    'muscle_increase_factor': 1,
    'babble_max_prob': 0.005,
    'hebbian_gain': 1,
}

run_id = int(time.time())
log_dir = (
    f'./{log_top_dir}/{run_id}'
    f'-W0-{options['W0']}-lr-{options['learning_rate']}'
    f'-damp-{options['damping']}-visc-{options['viscosity']}'
    f'-seed-{options['seed']}'
    f'-musc-stren-{options['muscle_strength_factor']}'
    f'-bab-prob-{options['babble_max_prob']}'
    f'-heb-gain-{options['hebbian_gain']}'
)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

sim_animation = f'./{log_dir}/animations/sim'
weight_animation = f'./{log_dir}/animations/weights'
weight_diff_animation = f'./{log_dir}/animations/weights_diff'
sim_log = f'./{log_dir}/data/sim-data'
heb_log = f'./{log_dir}/data/heb-data'

if not os.path.exists(f'./{log_dir}/data'):
    os.makedirs(f'./{log_dir}/data')
if not os.path.exists(f'./{log_dir}/weights'):
    os.makedirs(f'./{log_dir}/weights')

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
    'task': 'babble',
    'path_to_logfile': sim_log,
    'scope_port': env_scope_port,
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
    'Ia_velocity_gain': 8,
    'muscle_bias': -0.3,
    'muscle_gain': 1 / 0.7,
    'W_Ia':  W_Ia0,
    'K': 0.3,
    'k_leak': 0.2
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

spinal_layer = IaReflexSpinalLayer(**spinal_kwargs)
env = Arm26Babble(spinal_layer=spinal_layer,**env_kwargs)
learner = HebbianLearner(
    W=spinal_layer.net.aMN.W_Ia,
    **hebbian_kwargs
)

udpcast = scope.ScopeBroadcaster(port=scope_port)

filecast = scope.Scope

simulation_timestep = env.model.opt.timestep
env.model.opt.gravity[:] = [0.0, 0.0, 0.0]
env.model.opt.viscosity = options['viscosity']
env.model.dof_damping[:] = options['damping']

env.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
env.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = False
env.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
env.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True

mn_labels = ['mnSF', 'mnSE', 'mnEF', 'mnEE', 'mnBF', 'mnBE']
ia_labels = ['IaSF', 'IaSE', 'IaEF', 'IaEE', 'IaBF', 'IaBE']

monitor_W = weight_monitor.WeightMonitor(
    W=spinal_layer.net.aMN.W_Ia,
    save_frames_to_path=weight_animation,
    mn_labels=mn_labels,
    ia_labels=ia_labels,
    title=f'Eps: {int(env.eps)}, Time: {env.data.time:.1f} s'
)

monitor_diffW = weight_monitor.WeightMonitor(
    W=spinal_layer.net.aMN.W_Ia*0,
    save_frames_to_path=weight_diff_animation,
    mn_labels=mn_labels,
    ia_labels=ia_labels,
    scale_opt=1,
    title=f'Eps: {int(env.eps)}, Time: {env.data.time:.1f} s'
)


env.reset()
env.set_logfile(f'{sim_log}-{env.eps}')
learner.set_logfile(f'{heb_log}-{env.eps}')
muscle_strength_factor = options['muscle_strength_factor']
try:
    for _ in range(options['num_eps']):
        done = False
        tprv = 0
        cnt = 0
        while not done:
            _, _, done, _, _ = env.step(action=None)
            learner.step(
                pre_synaptic=spinal_layer.act_Ia,
                post_synaptic=spinal_layer.act_aMN
            )
            udpcast.broadcast(learner.data_cache)
            cnt += 1
            if cnt % (options['save_W_period']/simulation_timestep) == 0:
                print(
                    f'Eps: {int(env.eps)}, Time: {env.data.time:.1f} s,'
                    f'Muscle strength: {muscle_strength_factor}'
                )
                # Save to pickle file
                with open(f'{log_dir}/weights/W-{int(env.data.time)}', 'wb') as f:
                    pickle.dump(spinal_layer.net.aMN.W_Ia, f)
                monitor_W.update_plot(
                    spinal_layer.net.aMN.W_Ia,
                    title=f'Eps: {int(env.eps)}, Time: {env.data.time:.1f} s'
                )
                monitor_diffW.update_plot(
                    spinal_layer.net.aMN.W_Ia - W_Ia0,
                    title=f'Eps: {int(env.eps)}, Time: {env.data.time:.1f} s'
                )
        env.set_logfile(f'{sim_log}-{env.eps}')
        learner.set_logfile(f'{heb_log}-{env.eps}')
        env.reset()
        muscle_strength_factor = (
            muscle_strength_factor * options['muscle_increase_factor']
        )
        env.set_muscle_strength_factor(muscle_strength_factor)
        #with open(f'{log_dir}/learner-{int(env.data.time)}', 'wb') as f:
        #    pickle.dump(learner, f)

except Exception as e:
    tb = traceback.format_exc()
    print(f"Error during simulation: {str(e)}")
    print(f"Traceback details:\n{tb}")
finally:
    env.close()

