import torch
from deep_spine import (
    Arm26, LogCallbackArm26,
    Arm26Babble, LogCallbackArm26Babble,
    IdentitySpinalLayer,
    IaReflexRecInhPolicy,
    IaReflexPolicy
)

from stable_baselines3 import PPO, SAC
import deep_spine.utils.utils as utils


def initialize_configuration(settings, new_kwargs=None):
    config = {}
    config['settings'] = settings
    config['spinal_layer'] = spinal_layers[settings['spinal_layer']]
    config['supraspinal_layer'] = supraspinal_layers[
        settings['supraspinal_layer']
    ]

    config['training_algorithm'] = training_algorithms[
        config['supraspinal_layer']['algorithm']
    ]

    config['environment'] = environments[settings['environment']]

    simulation_steps_per_episode = (
            config['environment']['kwargs']['episode_length']
            * settings['fs_mujoco']
    )
    training_steps_per_episode = (
        simulation_steps_per_episode
        / config['environment']['kwargs']['supraspinal_decimation_factor']
    )
    training_steps_per_checkpoint = (
        training_steps_per_episode * settings['episodes_per_checkpoint']
    )
    total_training_steps = (
        settings['training_episodes'] * training_steps_per_episode
    )
    learning_rate_cyclic_period = (
        training_steps_per_checkpoint / total_training_steps
    )
    config['settings']['total_training_steps'] = total_training_steps
    config['settings']['training_steps_per_checkpoint'] = (
        training_steps_per_checkpoint
    )
    config['settings']['learning_rate_cyclic_period'] = (
        learning_rate_cyclic_period
    )

    if new_kwargs:
        try:
            config = utils.update_nested_dict(config, new_kwargs)
        except ConfigKeyError as e:
            raise ConfigKeyError(f"Invalid configiguration: {str(e)}")

    return config


supraspinal_layers = {
    'MlpPolicy-On-Policy': {
        'policy': 'MlpPolicy',
        'algorithm': 'PPO',
        'kwargs': {
            'activation_fn': torch.nn.Tanh,
            'net_arch': {
                'pi': [32,32],
                'vf': [32,32]
            }
        }
    },
    'MlpPolicy-Off-Policy': {
        'policy': 'MlpPolicy',
        'algorithm': 'SAC',
        'kwargs': {
            'activation_fn': torch.nn.Tanh,
            'net_arch': {
                'pi': [32,32],
                'qf': [32,32]
            }
        }
    },
    'IaReflexRecInhPolicy': {
        'policy': IaReflexRecInhPolicy,
        'algorithm': 'PPO',
        'kwargs': {
            'num_motor_units': 6,
            'num_proprioceptive_features': 12,
            'num_encoding_features': 4,
            'nn_size_critic': 32,
            'rec_inh_recurrent_network_structure': 'aa',
            'rec_inh_projection_network_structure': 'aa'
        }
    },
    'IaReflexPolicy': {
        'policy': IaReflexPolicy,
        'algorithm': 'PPO',
        'kwargs': {
            'num_motor_units': 6,
            'II_length_gain': 1.25,
            'II_bias': -0.25,
            'Ia_II_gain': 0.2,
            'Ia_saturation_constant': 1.5,
            'MN_bias': -0.1,
            'nn_size_critic': 32
        }
    },
    'pt-ppo': {
        'policy': 'pretrained',
        'algorithm': 'PPO',
    }
}

spinal_layers = {
    'identity': {
        'layer': IdentitySpinalLayer,
        'kwargs': {
            'afferent_mode': 'len-vel',
            'decimation_factor':  1,
            'afferent_filter_cutoff':  0.0,
        }
    }
}


training_algorithms = {
    'PPO': {
        'algorithm': PPO,
        'kwargs': {
            'batch_size': 2048,
            'n_epochs': 100,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'target_kl': None,
            'use_sde': False,
            'verbose': 0
        }
    },
    'SAC': {
        'algorithm': SAC,
        'kwargs': {
            'batch_size': 2048,
            'train_freq': (1, "episode"),
            'buffer_size': 1000000,
            'tau': 0.005,
            'ent_coef': 'auto',
            'gamma': 0.99,
            'target_update_interval': 1,
            'learning_starts': 1000,
            'use_sde': True,
            'use_sde_at_warmup': True,
            'sde_sample_freq': 64,
            'verbose': 0,
            'gradient_steps': -1
        }
    }
}


environments = {
    'arm26': {
        'environment': Arm26,
        'kwargs': {
            'task': 'static-pose',
            'episode_length': 10,
            'frame_rate': None,
            'save_frames_to_path': None,
            'max_noise_sigma': 0.1,
            'actuator_penalty':  0,
            'velocity_penalty':  0,
            'supraspinal_decimation_factor':  1,
        },
        'log_callback': LogCallbackArm26
    },
    'arm26-babble': {
        'environment': Arm26Babble,
        'kwargs': {
            'task': 'babble',
            'episode_length': 20,
            'xcor_window': 20000,
            'frame_rate': None,
            'save_frames_to_path': None,
            'supraspinal_decimation_factor':  1,
        },
        'log_callback': LogCallbackArm26Babble
    }
}


