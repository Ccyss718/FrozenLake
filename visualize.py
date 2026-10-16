import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

def moving_average(values, window_size):
    value_arr = np.array(values, dtype=np.float32)
    if window_size <=0:
        raise ValueError("window_size must be bigger than 0")
    elif window_size > len(value_arr):
        raise ValueError("window_size cannot exceed the data length")

    avg_res = []

    for i in range (len(value_arr) - window_size + 1):
        window = value_arr[i: i+window_size]
        avg = np.mean(window)
        avg_res.append(avg)

    return np.asarray(avg_res, dtype=float)

def plot_training_curve(history, output_path, window_size = 500):
    rewards = np.asarray(history["rewards"], dtype=float)
    success_rate = moving_average(rewards, window_size)

    episode_numbers = np.arange(window_size, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episode_numbers, success_rate)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate")
    ax.set_title("Training success rate")
    ax.set_ylim((0,1))
    ax.grid(visible=True)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path

def plot_evaluation_comparison(summary_report, output_path):
    group_lookup = {}

    for group in summary_report["groups"]:
        key = (group["environment_mode"], group["algorithm"])
        group_lookup[key] = group

    environment_modes = ["deterministic", "slippery"]
    q_means = []
    q_stds = []
    sarsa_means = []
    sarsa_stds = []

    for environment_mode in environment_modes:
        q_metric = group_lookup[
            (environment_mode, "q_learning")
        ]["metrics"]["evaluation_success_rate"]
        sarsa_metric = group_lookup[
            (environment_mode, "sarsa")
        ]["metrics"]["evaluation_success_rate"]

        q_means.append(q_metric["mean"])
        q_stds.append(q_metric["sample_std"])
        sarsa_means.append(sarsa_metric["mean"])
        sarsa_stds.append(sarsa_metric["sample_std"])

    x = np.arange(len(environment_modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        q_means,
        width,
        yerr=q_stds,
        label="Q-learning",
        capsize=5,
    )
    ax.bar(
        x + width / 2,
        sarsa_means,
        width,
        yerr=sarsa_stds,
        label="SARSA",
        capsize=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["Deterministic", "Slippery"])
    ax.set_xlabel("Environment")
    ax.set_ylabel("Evaluation success rate")
    ax.set_title("Q-learning vs SARSA")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path
