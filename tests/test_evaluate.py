import numpy as np

from evaluate import evaluate_policy


class EvaluationEnv:
    def __init__(self, transition):
        self.transition = transition
        self.actions = []
        self.reset_seeds = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return 0, {}

    def step(self, action):
        self.actions.append(action)
        return self.transition


def test_evaluation_is_greedy_and_does_not_change_q_table():
    env = EvaluationEnv((0, 1.0, True, False, {}))
    q_table = np.array([[0.1, 0.9, 0.2, 0.3]])
    before = q_table.copy()

    history = evaluate_policy(
        env,
        q_table,
        num_episodes=3,
        max_steps=10,
        seed=42,
    )

    assert env.actions == [1, 1, 1]
    assert env.reset_seeds == [42, None, None, None]
    assert history["rewards"] == [1.0, 1.0, 1.0]
    assert history["steps"] == [1, 1, 1]
    np.testing.assert_array_equal(q_table, before)


def test_evaluation_stops_when_the_environment_truncates():
    env = EvaluationEnv((0, 0.0, False, True, {}))
    q_table = np.array([[1.0, 0.0, 0.0, 0.0]])

    history = evaluate_policy(env, q_table, 1, max_steps=10, seed=42)

    assert len(env.actions) == 1
    assert history["terminated"] == [False]
    assert history["truncated"] == [True]
    assert history["steps"] == [1]


def test_evaluation_respects_max_steps():
    env = EvaluationEnv((0, 0.0, False, False, {}))
    q_table = np.array([[1.0, 0.0, 0.0, 0.0]])

    history = evaluate_policy(env, q_table, 1, max_steps=3, seed=42)

    assert len(env.actions) == 3
    assert history["steps"] == [3]
    assert history["terminated"] == [False]
    assert history["truncated"] == [False]


def test_evaluation_tie_breaking_is_reproducible():
    env_a = EvaluationEnv((0, 0.0, False, False, {}))
    env_b = EvaluationEnv((0, 0.0, False, False, {}))
    q_table = np.zeros((1, 4), dtype=float)

    history_a = evaluate_policy(env_a, q_table, 2, max_steps=10, seed=42)
    history_b = evaluate_policy(env_b, q_table, 2, max_steps=10, seed=42)

    assert env_a.actions == env_b.actions
    assert history_a == history_b
