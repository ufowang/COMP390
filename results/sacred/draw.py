import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def get_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith('info.json'):
                files.append(os.path.join(r, file))
    return files


def plot_return(paths, title='', label='', show=False):
    all_t_envs = []
    all_return_means = []

    for file_path in paths:
        with open(file_path, "r") as f:
            data = json.load(f)
            all_t_envs.append(data["return_mean_T"])
            all_return_means.append([item["value"] for item in data["return_mean"]])

    min_length = min(len(x) for x in all_return_means)

    trimmed_t_envs = [x[:min_length] for x in all_t_envs]
    t_envs = trimmed_t_envs[0]

    trimmed_return_means = [x[:min_length] for x in all_return_means]

    return_means_array = np.array(trimmed_return_means, dtype=np.float64)
    return_mean = np.mean(return_means_array, axis=0)
    return_std = np.std(return_means_array, axis=0)

    plt.plot(t_envs, return_mean, label=label, linewidth=0.8)
    plt.fill_between(t_envs, return_mean - return_std, return_mean + return_std, alpha=0.3)

    if show:
        plt.title(title)
        plt.xlabel('Environment Time')
        plt.ylabel('Return')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()


def plot_episode_length(paths, title='', label='', show=False):
    all_t_envs = []
    all_return_means = []

    for file_path in paths:
        with open(file_path, "r") as f:
            data = json.load(f)
            all_t_envs.append(data["ep_length_mean_T"])
            all_return_means.append(data["ep_length_mean"])

    min_length = min(len(x) for x in all_return_means)

    trimmed_t_envs = [x[:min_length] for x in all_t_envs]
    t_envs = trimmed_t_envs[0]

    trimmed_return_means = [x[:min_length] for x in all_return_means]

    return_means_array = np.array(trimmed_return_means, dtype=np.float64)
    return_mean = np.mean(return_means_array, axis=0)
    return_std = np.std(return_means_array, axis=0)

    plt.plot(t_envs, return_mean, label=label, linewidth=0.8)
    plt.fill_between(t_envs, return_mean - return_std, return_mean + return_std, alpha=0.3)

    if show:
        plt.title(title)
        plt.xlabel('Environment Time')
        plt.ylabel('Episode length')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()


def plot_QMIX():
    QMIX = get_files('predator_prey/QMIX')
    plot_return(QMIX, 'QMIX', '', True)
    plot_episode_length(QMIX, 'QMIX', '', True)


def plot_QMIX_Dueling():
    QMIX = get_files('predator_prey/QMIX')
    QMIX_Dueling = get_files('predator_prey/Dueling')
    plot_return(QMIX, '', 'QMIX')
    plot_return(QMIX_Dueling, 'Dueling Network', 'QMIX + Dueling Network', True)
    plot_episode_length(QMIX, '', 'QMIX')
    plot_episode_length(QMIX_Dueling, 'Dueling Network', 'QMIX + Dueling Network', True)


def plot_QMIX_Multi_Step():
    QMIX = get_files('predator_prey/QMIX')
    QMIX_Multi_Step = get_files('predator_prey/Multi-step')
    plot_return(QMIX, '', 'QMIX')
    plot_return(QMIX_Multi_Step, 'Multi-step Learning', 'QMIX + Multi-step Learning', True)
    plot_episode_length(QMIX, '', 'QMIX')
    plot_episode_length(QMIX_Multi_Step, 'Multi-step Learning', 'QMIX + Multi-step Learning', True)


def plot_QMIX_Priority_Replay_Buffer():
    QMIX = get_files('predator_prey/QMIX')
    QMIX_Priority_Replay_Buffer = get_files('predator_prey/Priority Replay Buffer 0.6 0.4 0.00001')
    plot_return(QMIX, '', 'QMIX')
    plot_return(QMIX_Priority_Replay_Buffer, 'Priority Replay', 'QMIX + Priority Replay', True)
    plot_episode_length(QMIX, '', 'QMIX')
    plot_episode_length(QMIX_Priority_Replay_Buffer, 'Priority Replay', 'QMIX + Priority Replay', True)


def plot_all():
    QMIX = get_files('predator_prey/QMIX')
    plot_return(QMIX, '', 'QMIX')
    QMIX_Dueling = get_files('predator_prey/Dueling')
    plot_return(QMIX_Dueling, '', 'QMIX + Dueling Network')
    QMIX_Multi_Step = get_files('predator_prey/Multi-step')
    plot_return(QMIX_Multi_Step, '', 'QMIX + Multi-step Learning')
    QMIX_Priority_Replay_Buffer = get_files('predator_prey/Priority Replay Buffer 0.6 0.4 0.00001')
    plot_return(QMIX_Priority_Replay_Buffer, 'Predator Prey', 'QMIX + Priority Replay', True)


def plot_all_episode_length():
    QMIX = get_files('predator_prey/QMIX')
    plot_episode_length(QMIX, '', 'QMIX')
    QMIX_Dueling = get_files('predator_prey/Dueling')
    plot_episode_length(QMIX_Dueling, '', 'QMIX + Dueling Network')
    QMIX_Multi_Step = get_files('predator_prey/Multi-step')
    plot_episode_length(QMIX_Multi_Step, '', 'QMIX + Multi-step Learning')
    QMIX_Priority_Replay_Buffer = get_files('predator_prey/Priority Replay Buffer 0.6 0.4 0.00001')
    plot_episode_length(QMIX_Priority_Replay_Buffer, 'Predator Prey', 'QMIX + Priority Replay', True)


# plot_QMIX_Dueling()
# plot_QMIX_Multi_Step()
# plot_QMIX_Priority_Replay_Buffer()
# plot_all()
# plot_all_episode_length()


def plot_Multi_Step():
    three_step = get_files('stag_hunt/punishment=0/multi_step/multi_step=3')
    five_step = get_files('stag_hunt/punishment=0/multi_step/multi_step=5')
    plot_return(three_step, '', 'Three-step')
    plot_return(five_step, 'Multi-step Learning', 'Five-step', True)
    plot_episode_length(three_step, '', 'Three-step')
    plot_episode_length(five_step, 'Multi-step Learning', 'Five-step', True)


# plot_Multi_Step()


def plot_Priority_Replay():
    sf4 = get_files('predator_prey/Priority Replay Buffer 0.6 0.4 0.0001')
    tf4 = get_files('predator_prey/Priority Replay Buffer 0.3 0.4 0.0001')
    sf5 = get_files('predator_prey/Priority Replay Buffer 0.6 0.4 0.00001')
    plot_return(sf4, '', 'Alpha 0.6, Epsilon 0.0001')
    plot_return(tf4, '', 'Alpha 0.3, Epsilon 0.0001')
    plot_return(sf5, 'Priority Replay', 'Alpha 0.6, Epsilon 0.00001', True)
    plot_episode_length(sf4, '', 'Alpha 0.6, Epsilon 0.0001')
    plot_episode_length(tf4, '', 'Alpha 0.3, Epsilon 0.0001')
    plot_episode_length(sf5, 'Priority Replay', 'Alpha 0.6, Epsilon 0.00001', True)

# plot_Priority_Replay()

def plot_hare():
    QMIX = get_files('predator_prey_hare/QMIX')
    Multi = get_files('predator_prey_hare/Multi')
    plot_return(QMIX, '', 'QMIX')
    plot_return(Multi, '4 hares & 1 Stag', 'QMIX + Multi-step Learning', True)
    plot_episode_length(QMIX, '', 'QMIX')
    plot_episode_length(Multi, '4 hares & 1 Stag', 'QMIX + Multi-step Learning', True)

# plot_hare()

def plot_reward_time():
    QMIX = get_files('predator_prey_reward_time/QMIX')
    Multi = get_files('predator_prey_reward_time/Multi')
    plot_return(QMIX, '', 'QMIX')
    plot_return(Multi, 'Reward Time 0', 'QMIX + Multi-step Learning', True)
    plot_episode_length(QMIX, '', 'QMIX')
    plot_episode_length(Multi, 'Reward Time 0', 'QMIX + Multi-step Learning', True)

# plot_reward_time()

