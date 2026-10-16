import numpy as np
import pytest

import train
from train import (
    linear_epsilon,
    train_one_episode,
    train_one_episode_sarsa,
    train_q_learning,
    train_sarsa,
)


class FakeSpace:
    def __init__(self, n):
        self.n = n


class OneStepTrainingEnv:
    """每回合一步获得奖励并终止，用于验证多回合训练。"""

    def __init__(self):
        self.observation_space = FakeSpace(1)
        self.action_space = FakeSpace(2)
        self.reset_seeds = []
        self.step_calls = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return 0, {}

    def step(self, action):
        self.step_calls += 1
        return 0, 1.0, True, False, {}


class FakeEpisodeEnv:
    """按给定顺序返回转移，用于验证单回合训练控制流。"""

    def __init__(self, initial_state, transitions):
        self.initial_state = initial_state
        self.transitions = list(transitions)
        self.step_index = 0
        self.actions = []

    def reset(self):
        self.step_index = 0
        self.actions.clear()
        return self.initial_state, {}

    def step(self, action):
        if self.step_index >= len(self.transitions):
            raise AssertionError("回合结束或达到最大步数后仍调用了step()")

        self.actions.append(action)
        transition = self.transitions[self.step_index]
        self.step_index += 1
        return transition


def test_one_step_termination_updates_the_selected_action():
    env = FakeEpisodeEnv(
        initial_state=0,
        transitions=[(1, 1.0, True, False, {})],
    )
    q_table = np.zeros((2, 4), dtype=float)
    q_table[0, 2] = 0.1

    result = train_one_episode(
        env,
        q_table,
        epsilon=0.0,
        alpha=1.0,
        gamma=0.9,
        rng=np.random.default_rng(42),
        max_steps=10,
    )

    assert result == (1.0, 1, True, False)
    assert env.actions == [2]
    assert q_table[0, 2] == pytest.approx(1.0)


def test_state_advances_between_steps():
    env = FakeEpisodeEnv(
        initial_state=0,
        transitions=[
            (1, 0.0, False, False, {}),
            (2, 1.0, True, False, {}),
        ],
    )
    q_table = np.zeros((3, 4), dtype=float)
    q_table[0, 1] = 0.5
    q_table[1, 2] = 0.8

    result = train_one_episode(
        env,
        q_table,
        epsilon=0.0,
        alpha=0.2,
        gamma=0.9,
        rng=np.random.default_rng(42),
        max_steps=10,
    )

    assert result == (1.0, 2, True, False)
    assert env.actions == [1, 2]


def test_truncation_stops_the_episode_but_keeps_bootstrapping():
    env = FakeEpisodeEnv(
        initial_state=0,
        transitions=[(1, -1.0, False, True, {})],
    )
    q_table = np.zeros((2, 4), dtype=float)
    q_table[0, 1] = 0.5
    q_table[1, 2] = 4.0

    result = train_one_episode(
        env,
        q_table,
        epsilon=0.0,
        alpha=0.2,
        gamma=0.5,
        rng=np.random.default_rng(42),
        max_steps=10,
    )

    assert result == (-1.0, 1, False, True)
    assert env.actions == [1]
    assert q_table[0, 1] == pytest.approx(0.6)


def test_max_steps_limits_environment_interactions():
    non_terminal_transition = (0, 0.0, False, False, {})
    env = FakeEpisodeEnv(
        initial_state=0,
        transitions=[non_terminal_transition] * 3,
    )
    q_table = np.zeros((1, 4), dtype=float)
    q_table[0, 0] = 0.1

    result = train_one_episode(
        env,
        q_table,
        epsilon=0.0,
        alpha=0.2,
        gamma=0.9,
        rng=np.random.default_rng(42),
        max_steps=2,
    )

    assert result == (0.0, 2, False, False)
    assert env.step_index == 2


def test_train_q_learning_runs_all_episodes_and_accumulates_updates():
    env = OneStepTrainingEnv()

    q_table, history = train_q_learning(
        env,
        num_episodes=3,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_episodes=1,
        alpha=0.5,
        gamma=0.9,
        max_steps=5,
        seed=42,
    )

    assert q_table.shape == (1, 2)
    assert env.step_calls == 3
    assert env.reset_seeds == [42, None, None, None]
    assert all(len(values) == 3 for values in history.values())
    assert history["epsilon"] == [0.0, 0.0, 0.0]
    assert history["rewards"] == [1.0, 1.0, 1.0]
    assert np.max(q_table) == pytest.approx(0.875)
    assert np.count_nonzero(q_table) == 1


def test_train_q_learning_is_reproducible_with_the_same_seed():
    env_a = OneStepTrainingEnv()
    env_b = OneStepTrainingEnv()
    arguments = {
        "num_episodes": 20,
        "epsilon_start": 0.3,
        "epsilon_end": 0.05,
        "epsilon_decay_episodes": 15,
        "alpha": 0.2,
        "gamma": 0.9,
        "max_steps": 5,
        "seed": 42,
    }

    q_table_a, history_a = train_q_learning(env_a, **arguments)
    q_table_b, history_b = train_q_learning(env_b, **arguments)

    np.testing.assert_array_equal(q_table_a, q_table_b)
    assert history_a == history_b


def test_train_q_learning_passes_each_scheduled_epsilon_to_the_episode(monkeypatch):
    env = OneStepTrainingEnv()
    received_epsilons = []

    def fake_train_one_episode(
        env,
        q_table,
        epsilon,
        alpha,
        gamma,
        rng,
        max_steps,
    ):
        received_epsilons.append(epsilon)
        return 0.0, 1, True, False

    monkeypatch.setattr(train, "train_one_episode", fake_train_one_episode)

    _, history = train_q_learning(
        env,
        num_episodes=4,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_episodes=3,
        alpha=0.2,
        gamma=0.9,
        max_steps=5,
        seed=42,
    )

    expected = [1.0, 0.7, 0.4, 0.1]
    assert received_epsilons == pytest.approx(expected)
    assert history["epsilon"] == pytest.approx(expected)


def test_sarsa_reuses_the_target_next_action_on_the_next_step(monkeypatch):
    env = FakeEpisodeEnv(
        initial_state=0,
        transitions=[
            (1, 0.0, False, False, {}),
            (2, 1.0, True, False, {}),
        ],
    )
    q_table = np.zeros((3, 4), dtype=float)
    actions_to_select = iter([1, 2, 3])
    selected_states = []
    update_calls = []

    def fake_select_action(q_table, state, epsilon, rng):
        selected_states.append(state)
        return next(actions_to_select)

    def fake_update_sarsa(
        q_table,
        state,
        action,
        reward,
        next_state,
        next_action,
        terminated,
        alpha,
        gamma,
    ):
        update_calls.append(
            (state, action, reward, next_state, next_action, terminated)
        )
        return 0.0

    monkeypatch.setattr(train.agent, "select_action", fake_select_action)
    monkeypatch.setattr(train.agent, "update_sarsa", fake_update_sarsa)

    result = train_one_episode_sarsa(
        env,
        q_table,
        epsilon=0.1,
        alpha=0.2,
        gamma=0.9,
        rng=np.random.default_rng(42),
        max_steps=10,
    )

    assert result == (1.0, 2, True, False)
    assert selected_states == [0, 1]
    assert env.actions == [1, 2]
    assert update_calls == [
        (0, 1, 0.0, 1, 2, False),
        (1, 2, 1.0, 2, None, True),
    ]


def test_sarsa_terminal_transition_updates_without_a_next_action():
    env = FakeEpisodeEnv(
        initial_state=0,
        transitions=[(1, 1.0, True, False, {})],
    )
    q_table = np.zeros((2, 4), dtype=float)
    q_table[0, 2] = 0.1

    result = train_one_episode_sarsa(
        env,
        q_table,
        epsilon=0.0,
        alpha=1.0,
        gamma=0.9,
        rng=np.random.default_rng(42),
        max_steps=10,
    )

    assert result == (1.0, 1, True, False)
    assert env.actions == [2]
    assert q_table[0, 2] == pytest.approx(1.0)


def test_sarsa_truncation_stops_but_uses_a_next_action_for_bootstrapping():
    env = FakeEpisodeEnv(
        initial_state=0,
        transitions=[(1, -1.0, False, True, {})],
    )
    q_table = np.zeros((2, 4), dtype=float)
    q_table[0, 1] = 0.5
    q_table[1, 2] = 4.0

    result = train_one_episode_sarsa(
        env,
        q_table,
        epsilon=0.0,
        alpha=0.2,
        gamma=0.5,
        rng=np.random.default_rng(42),
        max_steps=10,
    )

    assert result == (-1.0, 1, False, True)
    assert env.actions == [1]
    assert q_table[0, 1] == pytest.approx(0.6)


def test_train_sarsa_runs_all_episodes_and_accumulates_updates():
    env = OneStepTrainingEnv()

    q_table, history = train_sarsa(
        env,
        num_episodes=3,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_episodes=1,
        alpha=0.5,
        gamma=0.9,
        max_steps=5,
        seed=42,
    )

    assert q_table.shape == (1, 2)
    assert env.step_calls == 3
    assert env.reset_seeds == [42, None, None, None]
    assert all(len(values) == 3 for values in history.values())
    assert history["epsilon"] == [0.0, 0.0, 0.0]
    assert history["rewards"] == [1.0, 1.0, 1.0]
    assert np.max(q_table) == pytest.approx(0.875)
    assert np.count_nonzero(q_table) == 1


def test_train_sarsa_is_reproducible_with_the_same_seed():
    env_a = OneStepTrainingEnv()
    env_b = OneStepTrainingEnv()
    arguments = {
        "num_episodes": 20,
        "epsilon_start": 0.3,
        "epsilon_end": 0.05,
        "epsilon_decay_episodes": 15,
        "alpha": 0.2,
        "gamma": 0.9,
        "max_steps": 5,
        "seed": 42,
    }

    q_table_a, history_a = train_sarsa(env_a, **arguments)
    q_table_b, history_b = train_sarsa(env_b, **arguments)

    np.testing.assert_array_equal(q_table_a, q_table_b)
    assert history_a == history_b


def test_train_sarsa_passes_each_scheduled_epsilon_to_the_episode(monkeypatch):
    env = OneStepTrainingEnv()
    received_epsilons = []

    def fake_train_one_episode_sarsa(
        env,
        q_table,
        epsilon,
        alpha,
        gamma,
        rng,
        max_steps,
    ):
        received_epsilons.append(epsilon)
        return 0.0, 1, True, False

    monkeypatch.setattr(
        train,
        "train_one_episode_sarsa",
        fake_train_one_episode_sarsa,
    )

    _, history = train_sarsa(
        env,
        num_episodes=4,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_episodes=3,
        alpha=0.2,
        gamma=0.9,
        max_steps=5,
        seed=42,
    )

    expected = [1.0, 0.7, 0.4, 0.1]
    assert received_epsilons == pytest.approx(expected)
    assert history["epsilon"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("episode", "expected"),
    [
        (0, 1.0),
        (1000, 0.7525),
        (2000, 0.505),
        (4000, 0.01),
        (5000, 0.01),
    ],
)
def test_linear_epsilon_reaches_and_respects_its_lower_bound(episode, expected):
    epsilon = linear_epsilon(
        episode=episode,
        epsilon_start=1.0,
        epsilon_end=0.01,
        decay_episodes=4000,
    )

    assert epsilon == pytest.approx(expected)
