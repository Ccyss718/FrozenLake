from types import SimpleNamespace

import numpy as np
import pytest

import evaluate_saved


class ClosableEnv:
    def __init__(self):
        self.observation_space = SimpleNamespace(n=16)
        self.action_space = SimpleNamespace(n=4)
        self.closed = False

    def close(self):
        self.closed = True


def test_main_loads_q_table_and_runs_independent_evaluation(monkeypatch, capsys):
    environment = ClosableEnv()
    q_table = np.zeros((16, 4), dtype=float)
    evaluation_call = {}

    def fake_load_q_table(file_path):
        assert file_path == evaluate_saved.config.Q_TABLE_PATH
        return q_table

    def fake_make(*args, **kwargs):
        assert kwargs == {
            "id": evaluate_saved.config.ENV_ID,
            "map_name": evaluate_saved.config.MAP,
            "is_slippery": evaluate_saved.config.IS_SLIDE,
        }
        return environment

    def fake_evaluate(env, loaded_q_table, num_episodes, max_steps, seed):
        evaluation_call.update(
            env=env,
            q_table=loaded_q_table,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
        )
        return {
            "rewards": [1.0, 1.0, 1.0, 0.0],
            "steps": [5, 6, 7, 6],
        }

    monkeypatch.setattr(evaluate_saved.artifacts, "load_q_table", fake_load_q_table)
    monkeypatch.setattr(evaluate_saved.gym, "make", fake_make)
    monkeypatch.setattr(
        evaluate_saved.evaluate,
        "evaluate_policy",
        fake_evaluate,
    )

    evaluate_saved.main()

    assert environment.closed
    assert evaluation_call["env"] is environment
    assert evaluation_call["q_table"] is q_table
    assert evaluation_call["num_episodes"] == evaluate_saved.config.EVALUATE_TIMES
    assert evaluation_call["max_steps"] == evaluate_saved.config.MAX_STEP
    assert evaluation_call["seed"] == evaluate_saved.config.EVALUATE_SEED

    output = capsys.readouterr().out
    assert "q_table_loaded_from:" in output
    assert "evaluate_success_rate: 0.75" in output
    assert "average_steps: 6.0" in output


def test_main_rejects_wrong_q_table_shape_and_closes_environment(monkeypatch):
    environment = ClosableEnv()
    wrong_q_table = np.zeros((2, 2), dtype=float)

    monkeypatch.setattr(
        evaluate_saved.artifacts,
        "load_q_table",
        lambda file_path: wrong_q_table,
    )
    monkeypatch.setattr(evaluate_saved.gym, "make", lambda *args, **kwargs: environment)

    def fail_if_evaluated(*args, **kwargs):
        raise AssertionError("A wrong-shaped Q-table must not be evaluated")

    monkeypatch.setattr(
        evaluate_saved.evaluate,
        "evaluate_policy",
        fail_if_evaluated,
    )

    with pytest.raises(ValueError) as error:
        evaluate_saved.main()

    assert error.value.args == ((2, 2), (16, 4))
    assert environment.closed


def test_main_closes_environment_when_evaluation_fails(monkeypatch):
    environment = ClosableEnv()
    q_table = np.zeros((16, 4), dtype=float)

    monkeypatch.setattr(
        evaluate_saved.artifacts,
        "load_q_table",
        lambda file_path: q_table,
    )
    monkeypatch.setattr(evaluate_saved.gym, "make", lambda *args, **kwargs: environment)

    def fail_evaluation(*args, **kwargs):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(
        evaluate_saved.evaluate,
        "evaluate_policy",
        fail_evaluation,
    )

    with pytest.raises(RuntimeError, match="evaluation failed"):
        evaluate_saved.main()

    assert environment.closed


def test_main_does_not_create_environment_when_loading_fails(monkeypatch):
    def fail_loading(file_path):
        raise FileNotFoundError(file_path)

    def fail_if_environment_is_created(*args, **kwargs):
        raise AssertionError("The environment must not be created after loading fails")

    monkeypatch.setattr(evaluate_saved.artifacts, "load_q_table", fail_loading)
    monkeypatch.setattr(
        evaluate_saved.gym,
        "make",
        fail_if_environment_is_created,
    )

    with pytest.raises(FileNotFoundError):
        evaluate_saved.main()
