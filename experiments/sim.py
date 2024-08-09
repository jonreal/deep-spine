import sys, getopt
import numpy as np
import argparse

from deep_spine import (
    Arm26,
    Arm26Babble,
    IdentitySpinalLayer,
    IaReflexRecInhPolicy
)

import deep_spine.utils as utils


env_classes = {
    'Arm26': Arm26,
    'Arm26Babble': Arm26Babble,
    # Add more environments here
}


def simulate(
    env_class = Arm26,
    agentpath = None,
    checkpoint = None,
    num_eps = 10,
    **env_kwargs
):
    if agentpath:
        rl_agent, spinal_layer, configuration = utils.load_agent(
            agentpath=agentpath,
            checkpoint=checkpoint
        )
    else:
        spinal_layer = IdentitySpinalLayer()

    env = env_class(spinal_layer=spinal_layer, **env_kwargs)

    observation, info = env.reset()
    if agentpath:
        action, _ = rl_agent.predict(observation)
    else:
        action = np.zeros(env.action_space.shape)
    try:
        for _ in range(num_eps):
            done = False
            while not done:
                observation, reward, done, truncated, info = env.step(action)
                if agentpath:
                    action, _ = rl_agent.predict(
                        observation, deterministic=True
                    )
            observation, info = env.reset()
            if agentpath:
                action, states = rl_agent.predict(
                    observation, deterministic=True
                )
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
    finally:
        env.close()

def get_env_specific_args(env_name):
    if env_name == 'Arm26':
        return {
            'task': {
                'default':'static-pose',
                'help':'Task for env Arm26'
            },
            'qd': {
                'default': [np.pi/4, np.pi/4],
                'help': 'Desired joint angles for static-pose task'
            },
            'max_noise_sigma': {
                'default': 0,
                'help': 'Maximum noise sigma'
            }
        }

    if env_name == 'Arm26Babble':
        return {
            'task': {
                'default':'babble',
                'help':'Task for env Arm26Babble'
            },
        }
    return {}

def parse_args():
    parser = argparse.ArgumentParser(description='Simulate environment.')
    parser.add_argument(
        '-e', '--env',
        choices=['Arm26', 'Arm26Babble'],
        required=False,
        help='Environment to simulate')
    parser.add_argument(
        '-m', '--path_to_model',
        required=False,
        help='Path to agent model'
    )
    parser.add_argument(
        '-c', '--checkpoint',
        required=False,
        help='Checkpoint to load'
    )
    parser.add_argument(
        '-n', '--num_eps',
        type=int,
        default=10,
        help='Number of episodes to simulate'
    )
    parser.add_argument(
        '-g', '--episode_length',
        type=int,
        default=10,
        help='Number of steps per episode'
    )
    parser.add_argument(
        '-l', '--path_to_logfile',
        required=False,
        help='Path to logfile'
    )
    parser.add_argument(
        '-v', '--save_frames_to_path',
        required=False,
        help='Path to save frames'
    )

    args, _ = parser.parse_known_args()
    env_args = get_env_specific_args(args.env)
    for arg_name, arg_kwargs in env_args.items():
        parser.add_argument(f'--{arg_name}', **arg_kwargs)

    if any(arg in ['-h', '--help'] for arg in sys.argv):
        parser.print_help()
        sys.exit()

    return parser.parse_args()


def main():

    args = parse_args()

    if args.path_to_model is None:
        args.path_to_model, args.checkpoint = None, None
    env_kwargs = {
        k: getattr(args, k)
            for k in get_env_specific_args(args.env) if hasattr(args, k)
    }

    env_kwargs['save_frames_to_path'] = args.save_frames_to_path
    env_kwargs['frame_rate'] = 30
    env_kwargs['episode_length'] = args.episode_length
    env_kwargs['num_eps'] = args.num_eps

    if args.env:
        env_class = env_classes[args.env]
    else:
        env_class = Arm26

    for key, value in env_kwargs.items():
        print(f"{key}: {value}")

    simulate(
        env_class=env_class,
        agentpath=args.path_to_model,
        checkpoint=args.checkpoint,
        **env_kwargs
    )

if __name__=="__main__":
  main()
