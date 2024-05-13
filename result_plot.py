#!/home/airlab/anaconda3/envs/FormationControl/bin/python
import argparse
import numpy as np
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt 


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='results/td3/v0', help='log directory')

    arg = parser.parse_args()
    log_dir = arg.log_dir

    # Helper from the library
    results_plotter.plot_results(
        [log_dir], 1e5, results_plotter.X_TIMESTEPS, "TD3 Triangle Formation"
    )
    plot_results(log_dir)

