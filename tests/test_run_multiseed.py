import pytest

import config
import run_multiseed


def make_fake_result(algorithm, is_slippery, train_seed, evaluate_seed):
    base = train_seed / 10000
    success_rate = base + (0.1 if is_slippery else 0.0)
    return {
        "algorithm": algorithm,
        "environment_mode": "slippery" if is_slippery else "deterministic",
        "is_slippery": is_slippery,
        "train_seed": train_seed,
        "evaluate_seed": evaluate_seed,
        "run_dir": "run",
        "q_table_path": "q.npy",
        "config_path": "config.json",
        "metrics_path": "metrics.json",
        "metrics": {
            "training": {"last_500_success_rate": success_rate},
            "evaluation": {
                "success_rate": success_rate,
                "average_reward": success_rate,
                "average_steps": 10.0 + base,
                "successful_average_steps": 8.0 + base,
            },
        },
    }


def test_summarize_values_uses_sample_standard_deviation():
    summary = run_multiseed.summarize_values([1.0, 2.0, 3.0])

    assert summary == {
        "values": [1.0, 2.0, 3.0],
        "mean": 2.0,
        "sample_std": 1.0,
    }


def test_summarize_values_handles_no_successful_episodes():
    assert run_multiseed.summarize_values([]) == {
        "values": [],
        "mean": None,
        "sample_std": None,
    }


def test_run_all_experiments_runs_twelve_isolated_seed_combinations(
    monkeypatch,
):
    calls = []
    saves = []

    def fake_run_experiment(**kwargs):
        calls.append(kwargs)
        return make_fake_result(
            kwargs["algorithm"],
            kwargs["is_slippery"],
            kwargs["train_seed"],
            kwargs["evaluate_seed"],
        )

    monkeypatch.setattr(
        run_multiseed.experiment,
        "run_experiment",
        fake_run_experiment,
    )
    monkeypatch.setattr(
        run_multiseed.artifacts,
        "save_json",
        lambda data, path: saves.append((data, path)),
    )

    runs_report, summary_report = run_multiseed.run_all_experiments()

    assert len(calls) == 12
    assert len(runs_report["runs"]) == 12
    assert len(summary_report["groups"]) == 4
    assert all(group["num_seeds"] == 3 for group in summary_report["groups"])
    assert {
        (call["algorithm"], call["is_slippery"], call["train_seed"])
        for call in calls
    } == {
        (algorithm, is_slippery, seed)
        for algorithm in config.ALGORITHMS
        for is_slippery in config.SLIPPERY_OPTIONS
        for seed in config.MULTI_SEEDS
    }
    assert all(
        call["evaluate_seed"]
        == call["train_seed"] + config.EVALUATE_SEED_OFFSET
        for call in calls
    )
    assert saves[0][1] == config.MULTISEED_RUNS_PATH
    assert saves[1][1] == config.MULTISEED_SUMMARY_PATH

    deterministic_q = next(
        group
        for group in summary_report["groups"]
        if group["algorithm"] == "q_learning"
        and not group["is_slippery"]
    )
    expected_values = [seed / 10000 for seed in config.MULTI_SEEDS]
    assert deterministic_q["metrics"]["evaluation_success_rate"][
        "values"
    ] == pytest.approx(expected_values)
