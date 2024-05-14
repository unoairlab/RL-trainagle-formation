#!/home/airlab/anaconda3/envs/FormationControl/bin/python
import gymnasium as gym
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from Triangle import TriangleEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv
import numpy as np

width, height = 4, 5
box_min = (0, 0)
box_max = (width, height)
max_side_length = 2


# Define a function to create a single instance of the custom environment
def make_custom_env():
    return TriangleEnv(max_side_length, box_min, box_max)

# Define a function to create a vectorized environment with multiple instances of the custom environment


def main(args):
    # Create a vectorized environment with 4 instances of the custom environment
    env_count = 4
    vec_env = DummyVecEnv([make_custom_env for _ in range(env_count)])

    # vec_env =DummyVecEnv([TriangleEnv(max_side_length, box_min, box_max) for _ in range(args.num_envs)])

    weight_path = os.path.join('results', args.model, args.version, 'best_model.zip')
    model = PPO.load(weight_path)

    obs = vec_env.reset()
    print(obs.shape)
    while True:

        try:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            vec_env.envs[0].render(mode="human")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ppo', choices=['ppo', 'td3'])
    parser.add_argument('--version', type=str,  required=True, help='version')
    parser.add_argument('--num_envs', type=int, default=4)

    args = parser.parse_args()
    main(args)