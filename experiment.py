import gymnasium as gym
import numpy as np

import artifacts
import config
import evaluate
import train
import visualize


def get_train_function(algorithm):
    if algorithm == "q_learning":
        return train.train_q_learning
    if algorithm == "sarsa":
        return train.train_sarsa
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def build_experiment_config(
    algorithm,
    is_slippery,
    train_seed,
    evaluate_seed,
    map_name,
):
    return {
        "experiment_tag": config.EXPERIMENT_TAG,
        "algorithm": algorithm,
        "environment": {
            "env_id": config.ENV_ID,
            "map_name": map_name,
            "is_slippery": is_slippery,
        },
        "training": {
            "epsilon_schedule": {
                "type": config.EPSILON_SCHEDULE_TYPE,
                "start": config.EPSILON_START,
                "end": config.EPSILON_END,
                "decay_episodes": config.EPSILON_DECAY_EPISODES,
            },
            "num_episodes": config.TRAINING_TIMES,
            "alpha": config.ALPHA,
            "gamma": config.GAMMA,
            "max_steps": config.MAX_STEP,
            "seed": train_seed,
        },
        "evaluation": {
            "num_episodes": config.EVALUATE_TIMES,
            "max_steps": config.MAX_STEP,
            "epsilon": 0.0,
            "seed": evaluate_seed,
        },
        "visualization": {
            "training_curve_window_size": config.CURVE_WINDOW_SIZE,
        },
    }


def calculate_metrics(train_history, evaluate_history):
    train_rewards = np.asarray(train_history["rewards"], dtype=float)
    evaluate_rewards = np.asarray(evaluate_history["rewards"], dtype=float)
    evaluate_steps = np.asarray(evaluate_history["steps"], dtype=float)
    terminated = np.asarray(evaluate_history["terminated"], dtype=bool)
    truncated = np.asarray(evaluate_history["truncated"], dtype=bool)

    success_mask = evaluate_rewards > 0
    success_count = int(np.sum(success_mask))
    successful_average_steps = (
        float(np.mean(evaluate_steps[success_mask]))
        if success_count > 0
        else None
    )

    return {
        "training": {
            "num_episodes": len(train_rewards),
            "last_500_success_rate": float(np.mean(train_rewards[-500:])),
        },
        "evaluation": {
            "num_episodes": len(evaluate_rewards),
            "success_rate": float(np.mean(evaluate_rewards)),
            "average_reward": float(np.mean(evaluate_rewards)),
            "average_steps": float(np.mean(evaluate_steps)),
            "successful_average_steps": successful_average_steps,
            "success_count": success_count,
            "failure_count": int(len(evaluate_rewards) - success_count),
            "terminated_count": int(np.sum(terminated)),
            "truncated_count": int(np.sum(truncated)),
        },
    }


def run_experiment(
    algorithm,
    is_slippery,
    train_seed,
    evaluate_seed,
    map_name=config.MAP,
):
    run_dir = config.build_run_dir(
        algorithm=algorithm,
        is_slippery=is_slippery,
        train_seed=train_seed,
        map_name=map_name,
    )
    curve_path = run_dir / "curves" / "training_success_rate.png"
    q_table_path = run_dir / "checkpoints" / "q_table.npy"
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"

    train_env = gym.make(
        id=config.ENV_ID,
        map_name=map_name,
        is_slippery=is_slippery,
    )
    try:
        train_function = get_train_function(algorithm)
        q_table, train_history = train_function(
            train_env,
            config.TRAINING_TIMES,
            config.EPSILON_START,
            config.EPSILON_END,
            config.EPSILON_DECAY_EPISODES,
            config.ALPHA,
            config.GAMMA,
            config.MAX_STEP,
            train_seed,
        )
    finally:
        train_env.close()

    visualize.plot_training_curve(
        train_history,
        curve_path,
        window_size=config.CURVE_WINDOW_SIZE,
    )

    experiment_config = build_experiment_config(
        algorithm=algorithm,
        is_slippery=is_slippery,
        train_seed=train_seed,
        evaluate_seed=evaluate_seed,
        map_name=map_name,
    )
    artifacts.save_q_table(q_table, q_table_path)
    artifacts.save_json(experiment_config, config_path)

    evaluate_env = gym.make(
        id=config.ENV_ID,
        map_name=map_name,
        is_slippery=is_slippery,
    )
    try:
        evaluate_history = evaluate.evaluate_policy(
            evaluate_env,
            q_table,
            config.EVALUATE_TIMES,
            config.MAX_STEP,
            evaluate_seed,
        )
    finally:
        evaluate_env.close()

    metrics = calculate_metrics(train_history, evaluate_history)
    artifacts.save_json(metrics, metrics_path)

    environment_mode = "slippery" if is_slippery else "deterministic"
    return {
        "algorithm": algorithm,
        "environment_mode": environment_mode,
        "is_slippery": is_slippery,
        "train_seed": train_seed,
        "evaluate_seed": evaluate_seed,
        "run_dir": str(run_dir),
        "q_table_path": str(q_table_path),
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "training_curve_path": str(curve_path),
    }
