import os, pickle, time
import numpy as np
from stable_baselines3 import PPO, SAC

from stable_baselines3.common.callbacks import BaseCallback

def create_label(params):
    label_parts = []
    def get_leaf_values(d):
        for key, value in d.items():
            if isinstance(value, dict):
                yield from get_leaf_values(value)
            else:
                yield key, value
    for key, value in get_leaf_values(params):
        label_parts.append(f"{key[:3]}_{value}")
    return "-".join(label_parts)

def update_nested_dict(base_dict, update_dict, path=None):
    if path is None:
        path = []

    for key, value in update_dict.items():
        if key not in base_dict:
            full_path = '.'.join(path + [key])
            raise ConfigKeyError(
                f"Key '{full_path}'"
                "does not exist in the default configuration."
            )
        if isinstance(value, dict) and isinstance(base_dict[key], dict):
            update_nested_dict(base_dict[key], value, path + [key])
        else:
            base_dict[key] = value
    return base_dict

def load_agent(agentpath, checkpoint=None):

    with open(f'{agentpath}/spinal_layer.pkl','rb') as f:
        spinal_layer = pickle.load(f)

    with open(f'{agentpath}/configuration.pkl','rb') as f:
        configuration = pickle.load(f)


    if checkpoint is None:
        allagents = get_all_agent_checkpoints(agentpath)
        runid = allagents[-1].split(".")
        runid = runid[0]
    else:
        runid = checkpoint + '.zip'

    print(f'runid = {runid}')
    training_algorithm = configuration['supraspinal_layer']['algorithm']
    path_to_model = f'{agentpath}/{runid}'
    if training_algorithm == "PPO":
        rl_agent = PPO.load(path_to_model)
    elif training_algorithm == "SAC":
        rl_agent = SAC.load(path_to_model)
    return rl_agent, spinal_layer, configuration

def get_all_agent_checkpoints(agentpath):
    allagents = sorted(os.listdir(agentpath))
    allagents.sort(
        key=lambda fn: os.path.getmtime(os.path.join(agentpath, fn))
    )
    allagents = [file for file in allagents if file.endswith('.zip')]
    return allagents

class SaveModelAndReportProgressCallback(BaseCallback):
    def __init__(self,
        model_id,
        save_path,
        training_episodes,
        model,
        env
    ):
        self.model_id = model_id
        self.save_path = save_path
        self.training_episodes = training_episodes
        self.num_steps = 0
        self.prev_time = time.time()
        self.elapsed = 0
        self.model = model
        self.env = env
        self.n_calls = 0

    def _on_step(self) -> bool:
        now = time.time()
        prev_num_steps = self.num_steps
        self.num_steps = self.model.num_timesteps
        elapsed = now - self.prev_time
        self.prev_time = now
        self.model.save(f'{self.save_path}/{self.num_steps}')
        print(
            f'{self.model_id}: '
            f'elapsed time = {elapsed:.2f} s,'
            f' avg frame rate = '
            f' {int((self.num_steps-prev_num_steps)/elapsed)},'
            f' episode = {self.env.eps}/{self.training_episodes}'
            '\n'
        )
        return True

def variable_learning_rate(progress_remaining,start_lr, end_lr):
    return start_lr + progress_remaining * (end_lr - start_lr)

def cosine_learning_rate(progress_remaining, period, max_lr, min_lr):
    normalized_time = ((1 - progress_remaining) % period) / period
    return (
        min_lr + (max_lr - min_lr) / 2 * (1 + np.cos(np.pi * normalized_time))
    )
