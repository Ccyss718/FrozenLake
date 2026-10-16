import numpy as np
import pytest

from agent import select_action, update_q, update_sarsa


def test_select_action_exploits_the_largest_q_value():
    q_table = np.array([[0.1, 0.7, 0.2, 0.3]])
    rng = np.random.default_rng(42)

    action = select_action(q_table, state=0, epsilon=0.0, rng=rng)

    assert action == 1


def test_select_action_exploration_is_valid_and_reproducible():
    q_table = np.array([[0.1, 0.7, 0.2, 0.3]])
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)

    actions_a = [
        select_action(q_table, state=0, epsilon=1.0, rng=rng_a)
        for _ in range(20)
    ]
    actions_b = [
        select_action(q_table, state=0, epsilon=1.0, rng=rng_b)
        for _ in range(20)
    ]

    assert actions_a == actions_b
    assert all(0 <= action < q_table.shape[1] for action in actions_a)
    assert len(set(actions_a)) > 1


def test_select_action_randomly_breaks_ties_between_best_actions():
    q_table = np.array([[0.7, 0.1, 0.7, 0.2]])
    rng = np.random.default_rng(42)

    actions = [
        select_action(q_table, state=0, epsilon=0.0, rng=rng)
        for _ in range(100)
    ]

    assert set(actions) == {0, 2}


def test_update_q_bootstraps_from_the_best_next_action():
    q_table = np.zeros((3, 4), dtype=float)
    q_table[0] = [0.1, 2.0, 0.3, 0.4]
    q_table[1] = [1.0, 4.0, 3.0, 2.0]
    before = q_table.copy()

    td_error = update_q(
        q_table,
        state=0,
        action=1,
        reward=-1.0,
        next_state=1,
        terminated=False,
        alpha=0.2,
        gamma=0.5,
    )

    assert td_error == pytest.approx(-1.0)
    assert q_table[0, 1] == pytest.approx(1.8)

    before[0, 1] = q_table[0, 1]
    np.testing.assert_array_equal(q_table, before)


def test_update_q_does_not_bootstrap_after_termination():
    q_table = np.zeros((3, 4), dtype=float)
    q_table[0, 1] = 2.0
    q_table[2] = [100.0, 100.0, 100.0, 100.0]

    td_error = update_q(
        q_table,
        state=0,
        action=1,
        reward=1.0,
        next_state=2,
        terminated=True,
        alpha=0.2,
        gamma=0.5,
    )

    assert td_error == pytest.approx(-1.0)
    assert q_table[0, 1] == pytest.approx(1.8)


def test_update_sarsa_bootstraps_from_the_selected_next_action():
    q_table = np.zeros((3, 4), dtype=float)
    q_table[0, 1] = 1.0
    q_table[1] = [0.5, 4.0, 2.0, 1.0]

    td_error = update_sarsa(
        q_table,
        state=0,
        action=1,
        reward=2.0,
        next_state=1,
        next_action=1,
        terminated=False,
        alpha=0.1,
        gamma=0.5,
    )

    assert td_error == pytest.approx(3.0)
    assert q_table[0, 1] == pytest.approx(1.3)


def test_update_sarsa_uses_the_actual_next_action_instead_of_the_maximum():
    q_table = np.zeros((3, 4), dtype=float)
    q_table[0, 1] = 2.0
    q_table[1] = [1.0, 4.0, 3.0, 2.0]
    before = q_table.copy()

    td_error = update_sarsa(
        q_table,
        state=0,
        action=1,
        reward=-1.0,
        next_state=1,
        next_action=2,
        terminated=False,
        alpha=0.2,
        gamma=0.5,
    )

    assert td_error == pytest.approx(-1.5)
    assert q_table[0, 1] == pytest.approx(1.7)

    before[0, 1] = q_table[0, 1]
    np.testing.assert_array_equal(q_table, before)


def test_update_sarsa_does_not_access_next_action_after_termination():
    q_table = np.zeros((3, 4), dtype=float)
    q_table[0, 1] = 2.0
    q_table[2] = [100.0, 100.0, 100.0, 100.0]

    td_error = update_sarsa(
        q_table,
        state=0,
        action=1,
        reward=1.0,
        next_state=2,
        next_action=None,
        terminated=True,
        alpha=0.2,
        gamma=0.5,
    )

    assert td_error == pytest.approx(-1.0)
    assert q_table[0, 1] == pytest.approx(1.8)
