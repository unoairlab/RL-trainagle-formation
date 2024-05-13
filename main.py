#!/home/airlab/anaconda3/envs/FormationControl/bin/python
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from Triangle import TriangleEnv
from stable_baselines3.common.env_checker import check_env


import os 
import numpy as np 
import argparse
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, training_steps:int, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.training_steps = training_steps

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps} / {self.training_steps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True

width, height = 4, 5
box_min = (0, 0)
box_max = (width, height)
max_side_length = 2


def get_model(arg, env, action_noise, verbose=0):
    if args.model == 'ppo':
        return PPO('MlpPolicy', env, verbose=0)
    return TD3('MlpPolicy', env, action_noise=action_noise, verbose=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['ppo', 'td3'])
    parser.add_argument('--log-dir', type=str, default='results/td3/v0', help='specify training log dir')
    parser.add_argument('--training-steps', type=int, default=5e4, help='number of training steps' )
    args = parser.parse_args()
    # Create log dir
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = TriangleEnv(max_side_length, box_min, box_max)
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)

        # Create action noise because TD3 and DDPG use a deterministic policy
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(args.training_steps, check_freq=1000, log_dir=log_dir)
    # Create RL model
    # model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0)
    model = get_model(args, env, action_noise) 
    # Train the agent
    model.learn(total_timesteps=args.training_steps, callback=callback)


