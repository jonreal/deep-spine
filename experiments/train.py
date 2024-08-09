import time
import os
import pickle
import torch
import copy
import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import (
    StopTrainingOnMaxEpisodes,
    CallbackList,
    EveryNTimesteps
)

from deep_spine import (
    Arm26,
    LogCallbackArm26,
    Arm26Babble,
    LogCallbackArm26Babble,
    IdentitySpinalLayer,
    IaReflexRecInhPolicy
)
import deep_spine.utils.utils as utils

import defaults

FS_MUJOCO = 200
tensorboard_eps_reward_moving_average_window = 10

pt_model_path = (
    './agents/tune-random-reflexes/models'
    + '/1720482881'
    + '/4000000'
)

settings = {
    'path_to_agents_dir': './agents/babbling-3',
    'supraspinal_layer': 'IaReflexPolicy',
    #   'pt_path': pt_model_path,
    'spinal_layer': 'identity',
    'environment': 'arm26-babble',
    'num_agents': 1,
    'training_episodes': 2000,
    'episodes_per_checkpoint': 500,
    'max_lr': 0.01,
    'min_lr': 0.001,
    'fs_mujoco':  FS_MUJOCO,
}

_kwargs = {
    'training_algorithm': {
        'kwargs': {'batch_size': 256}
    },
    'environment': {
        'kwargs': {
            'episode_length': 100,
            'xcor_window': 100
        }
    }
}

__kwargs = {
    'environment': {
        'kwargs': {
            'frame_rate': 30
        }
    }
}

configuration  = defaults.initialize_configuration(
    settings, {**_kwargs, **__kwargs}
)
label = utils.create_label(_kwargs)

print(configuration)


if settings['num_agents'] == 1:
    progress_bar = True
else:
    progress_bar = False

def train_agent(seed):
    time.sleep(2*configuration['settings']['num_agents'])

    spinal_layer = configuration['spinal_layer']['layer'](
        **configuration['spinal_layer']['kwargs']
    )

    model_id = int(time.time())
    path_to_agents_dir = configuration['settings']['path_to_agents_dir']
    mdir = f'./{path_to_agents_dir}/models/{model_id}'

    if configuration['supraspinal_layer']['policy'] == 'pretrained':
        supraspinal_layer_str = (
            f'-pt_{'_'.join(
                configuration['settings']['pt_path'].split('/')[-2:])}'
        )
    else:
        supraspinal_layer_str = f'-{settings['supraspinal_layer']}'

    ldir =  (
        f'./{path_to_agents_dir}/logs/{model_id}'
        f'-{spinal_layer.type}'
        f'{supraspinal_layer_str}'
        f'-{configuration['supraspinal_layer']['algorithm']}'
        f'-{label}'
    )
    print(ldir)

    if not os.path.exists(mdir):
        os.makedirs(mdir)
    if not os.path.exists(ldir):
        os.makedirs(ldir)
    with open(mdir + '/spinal_layer.pkl', 'wb') as f:
        pickle.dump(spinal_layer, f)
    with open(mdir + '/configuration.pkl', 'wb') as f:
        pickle.dump(configuration, f)

    env = configuration['environment']['environment'](
            **configuration['environment']['kwargs'],
            spinal_layer=spinal_layer,
            seed=seed
    )

    learning_rate = (
        lambda progress_remaining: (
            utils.cosine_learning_rate(
                progress_remaining,
                configuration['settings']['learning_rate_cyclic_period'],
                configuration['settings']['max_lr'],
                configuration['settings']['min_lr']
            )
        )
    )

    # Load pretrained model
    if configuration['supraspinal_layer']['policy'] == 'pretrained':
        print(configuration['training_algorithm']['kwargs'])
        model = configuration['training_algorithm']['algorithm'].load(
            configuration['settings']['pt_path'],
            env,
            **configuration['training_algorithm']['kwargs'],
            learning_rate=learning_rate,
            seed = int(seed),
            tensorboard_log=ldir
        )
        model.num_timesteps = 0

    # Non pretrained model
    else:
        model = configuration['training_algorithm']['algorithm'](
            configuration['supraspinal_layer']['policy'],
            env,
            **configuration['training_algorithm']['kwargs'],
            policy_kwargs=configuration['supraspinal_layer']['kwargs'],
            learning_rate=learning_rate,
            seed=int(seed),
            tensorboard_log=ldir
        )

    print(model.policy)
    print(
        'Number of parameters: '
        f'{sum(p.numel() for p in model.policy.parameters())}'
    )
    model.save(f'{mdir}/{model.num_timesteps}')
    print(f'\n\nModel saved to: {mdir}/{model.num_timesteps}\n\n')

    callback = CallbackList(
        [
            StopTrainingOnMaxEpisodes(
                max_episodes=configuration['settings']['training_episodes']
            ),
            configuration['environment']['log_callback'](
                env,
                tensorboard_eps_reward_moving_average_window
            ),
            EveryNTimesteps(
                n_steps=(
                    configuration['settings']['training_steps_per_checkpoint']
                ),
                callback=utils.SaveModelAndReportProgressCallback(
                    model_id,
                    mdir,
                    configuration['settings']['training_episodes'],
                    model,
                    env
                )
            )
        ]
    )

    try:
        model.learn(
            total_timesteps=configuration['settings']['total_training_steps'],
            reset_num_timesteps=False,
            progress_bar=progress_bar,
            callback=callback
        )
    except ValueError as e:
        print(
            f"{model_id}: Training failed: "
            f"ValueError: {e}"
        )
    env.close()

if __name__ == '__main__':
    seed = np.arange(1, settings['num_agents'] + 1)
    processes = []
    if settings['num_agents'] > 1:
        for n in range(settings['num_agents']):
            p = torch.multiprocessing.Process(
                    target=train_agent, args=(int(seed[n]),)
            )
            p.start()
            processes.append(p)
            time.sleep(2)
        for p in processes:
            p.join()
    else:
        train_agent(int(seed[0]))
