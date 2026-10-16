import numpy as np
import pytest

import config
import experiment


class ClosableEnv:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_get_train_function_selects_supported_algorithms():
    assert experiment.get_train_function("q_learning") is experiment.train.train_q_learning
    assert experiment.get_train_function("sarsa") is experiment.train.train_sarsa


def test_get_train_function_rejects_an_unknown_algorithm():
    with pytest.raises(ValueError, match="Unsupported algorithm: unknown"):
        experiment.get_train_function("unknown")


def test_calculate_metrics_includes_success_and_end_reason_counts():
    train_history = {"rewards": [0.0, 1.0, 1.0]}
    evaluate_history = {
        "rewards": [1.0, 0.0, 1.0, 0.0],
        "steps": [6, 4, 8, 100],
        "terminated": [True, True, True, False],
        "truncated": [False, False, False, True],
    }

    metrics = experiment.calculate_metrics(train_history, evaluate_history)

    assert metrics["training"]["num_episodes"] == 3
    assert metrics["training"]["last_500_success_rate"] == pytest.approx(2 / 3)
    assert metrics["evaluation"] == {
        "num_episodes": 4,
        "success_rate": 0.5,
        "average_reward": 0.5,
        "average_steps": 29.5,
        "successful_average_steps": 7.0,
        "success_count": 2,
        "failure_count": 2,
        "terminated_count": 3,
        "truncated_count": 1,
    }


def test_run_experiment_separates_environments_and_saves_per_run_artifacts(
    monkeypatch,
):
    environments = []
    train_call = {}
    evaluate_call = {}
    plot_calls = []
    q_table_saves = []
    json_saves = []

    def fake_make(*args, **kwargs):
        environment = ClosableEnv()
        environments.append(environment)
        return environment

    def fake_train(
        env,
        num_episodes,
        epsilon_start,
        epsilon_end,
        epsilon_decay_episodes,
        alpha,
        gamma,
        max_steps,
        seed,
    ):
        train_call.update(
            env=env,
            num_episodes=num_episodes,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_episodes=epsilon_decay_episodes,
            alpha=alpha,
            gamma=gamma,
            max_steps=max_steps,
            seed=seed,
        )
        return np.zeros((16, 4)), {"rewards": [1.0] * 500}

    def fake_evaluate(env, q_table, num_episodes, max_steps, seed):
        evaluate_call.update(
            env=env,
            q_table=q_table,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
        )
        return {
            "rewards": [1.0] * num_episodes,
            "steps": [6] * num_episodes,
            "terminated": [True] * num_episodes,
            "truncated": [False] * num_episodes,
        }

    def fake_plot_training_curve(history, output_path, window_size):
        plot_calls.append(
            {
                "history": history,
                "output_path": output_path,
                "window_size": window_size,
            }
        )
        return output_path

    monkeypatch.setattr(experiment.gym, "make", fake_make)
    monkeypatch.setattr(experiment, "get_train_function", lambda _: fake_train)
    monkeypatch.setattr(
        experiment.evaluate,
        "evaluate_policy",
        fake_evaluate,
    )
    monkeypatch.setattr(
        experiment.visualize,
        "plot_training_curve",
        fake_plot_training_curve,
    )
    monkeypatch.setattr(
        experiment.artifacts,
        "save_q_table",
        lambda data, path: q_table_saves.append((data, path)),
    )
    monkeypatch.setattr(
        experiment.artifacts,
        "save_json",
        lambda data, path: json_saves.append((data, path)),
    )

    result = experiment.run_experiment(
        algorithm="sarsa",
        is_slippery=True,
        train_seed=123,
        evaluate_seed=10123,
        map_name="4x4",
    )

    expected_run_dir = config.build_run_dir(
        "sarsa",
        True,
        123,
        map_name="4x4",
    )
    assert len(environments) == 2
    assert environments[0] is not environments[1]
    assert all(environment.closed for environment in environments)
    assert train_call["env"] is environments[0]
    assert train_call["seed"] == 123
    assert evaluate_call["env"] is environments[1]
    assert evaluate_call["seed"] == 10123
    assert plot_calls == [
        {
            "history": {"rewards": [1.0] * 500},
            "output_path": expected_run_dir
            / "curves"
            / "training_success_rate.png",
            "window_size": config.CURVE_WINDOW_SIZE,
        }
    ]
    assert q_table_saves[0][1] == expected_run_dir / "checkpoints" / "q_table.npy"
    assert json_saves[0][1] == expected_run_dir / "config.json"
    assert json_saves[0][0]["training"]["seed"] == 123
    assert json_saves[0][0]["evaluation"]["seed"] == 10123
    assert json_saves[1][1] == expected_run_dir / "metrics.json"
    assert result["run_dir"] == str(expected_run_dir)
    assert result["training_curve_path"] == str(
        expected_run_dir / "curves" / "training_success_rate.png"
    )
    assert result["metrics"]["evaluation"]["success_rate"] == 1.0
