import numpy as np


def select_action(q_table, state, epsilon, rng):
    random_number = rng.random()
    action_count = q_table.shape[1]

    if random_number < epsilon:
        action = rng.integers(action_count)
    else:
        recent_q = q_table[state]
        max_qval = np.max(recent_q)
        best_action = np.flatnonzero(recent_q == max_qval)
        action = rng.choice(best_action)

    return int(action)

def update_q(
        q_table,
        state,
        action,
        reward,
        next_state,
        terminated,
        alpha,
        gamma,
):
    q_value = q_table[state, action]

    if terminated:
        target = reward
    else:
        next_bestq = np.max(q_table[next_state])
        target = reward + gamma*next_bestq

    td_error = target - q_value
    q_table[state, action] = q_value +alpha*td_error

    return td_error

def update_sarsa(
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
    q_value = q_table[state, action]
    if terminated:
        target = reward
    else:
        target = reward + gamma*q_table[next_state, next_action]

    error = target - q_value

    q_table[state, action] += alpha*error

    return error
     