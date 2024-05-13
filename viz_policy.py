#!/home/airlab/anaconda3/envs/FormationControl/bin/python
import gymnasium as gym
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from Triangle import TriangleEnv
from stable_baselines3.common.env_checker import check_env

width, height = 4, 5
box_min = (0, 0)
box_max = (width, height)
max_side_length = 2


def main(args):
    vec_env = TriangleEnv(max_side_length, box_min, box_max)

    weight_path = os.path.join('results', args.model, args.version, 'best_model.zip')
    model = PPO.load(weight_path)

    obs, _ = vec_env.reset()
    print(obs.shape)
    while True:

        try:
            action, _states = model.predict(obs)
            obs, rewards, dones, _, info = vec_env.step(action)
            vec_env.render("human")
        except KeyboardInterrupt:
            break 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ppo', choices=['ppo', 'td3'])
    parser.add_argument('--version', type=str, required=True, help='version')

    args = parser.parse_args()
    main(args)