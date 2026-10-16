from contextlib import redirect_stdout
from io import StringIO

import gymnasium as gym

import probe


class FakeSpace:
    def __init__(self, n, sampled_value=0):
        self.n = n
        self.sampled_value = sampled_value
        self.sample_calls = 0

    def sample(self):
        self.sample_calls += 1
        return self.sampled_value

    def __repr__(self):
        return f"Discrete({self.n})"


class FakeEnv:
    """固定完整回合的转移结果，检查probe的结束控制流。"""

    def __init__(self, episode_results):
        self.observation_space = FakeSpace(16)
        self.action_space = FakeSpace(4)
        self.episode_results = list(episode_results)
        self.reset_seeds = []
        self.episode_index = 0
        self.closed = False

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        if len(self.reset_seeds) == 2:
            self.episode_index = 0
        return 0, {"prob": 1.0}

    def step(self, action):
        if len(self.reset_seeds) == 1:
            return 1, 0.0, False, False, {"prob": 1.0}

        if self.episode_index >= len(self.episode_results):
            raise AssertionError("回合结束或达到最大步数后仍调用了step()")

        result = self.episode_results[self.episode_index]
        self.episode_index += 1
        return result

    def close(self):
        self.closed = True


def run_probe(fake_env, monkeypatch, max_steps=100):
    monkeypatch.setattr(probe.gym, "make", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(probe, "MAX_STEPS", max_steps)
    output = StringIO()

    with redirect_stdout(output):
        probe.main()

    return output.getvalue()


def test_frozenlake_spaces_match_the_4x4_map():
    env = gym.make(
        probe.ENV_ID,
        map_name=probe.MAP_NAME,
        is_slippery=False,
    )

    try:
        state, info = env.reset(seed=probe.SEED)

        assert env.observation_space.n == 16
        assert env.action_space.n == 4
        assert env.observation_space.contains(state)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_terminated_stops_the_episode(monkeypatch):
    env = FakeEnv([(5, 0.0, True, False, {})])

    output = run_probe(env, monkeypatch)

    assert "end_reason: terminated" in output
    assert "total_steps: 1" in output
    assert env.reset_seeds == [probe.SEED, probe.SEED]
    assert env.episode_index == 1
    assert env.closed is True


def test_truncated_stops_with_the_correct_reason(monkeypatch):
    env = FakeEnv([(1, 0.0, False, True, {})])

    output = run_probe(env, monkeypatch)

    assert "end_reason: truncated" in output
    assert "total_steps: 1" in output
    assert env.episode_index == 1
    assert env.closed is True


def test_max_steps_is_not_exceeded(monkeypatch):
    non_terminal = (0, 0.0, False, False, {})
    env = FakeEnv([non_terminal] * 5)

    output = run_probe(env, monkeypatch, max_steps=3)

    assert "end_reason: max_steps" in output
    assert "total_steps: 3" in output
    assert env.episode_index == 3
    assert env.closed is True
