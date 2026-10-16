import numpy as np
import agent

def train_one_episode(
        env,
        q_table,
        epsilon,
        alpha,
        gamma,
        rng,
        max_steps,
):
    state, info = env.reset()
    total_reward = 0
    terminated, truncated = False, False
    
    for steps in range(1, max_steps+1):
        action = agent.select_action(q_table, state, epsilon, rng)

        next_state, reward, terminated, truncated, info = env.step(action)

        td_error = agent.update_q(q_table, state, action, reward, next_state, terminated, alpha, gamma)
        total_reward += reward
        state = next_state

        if terminated or truncated:
            break

    return total_reward, steps, terminated, truncated

def train_q_learning(
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
    history = {
        "epsilon":[],
        "rewards":[],
        "steps":[],
        "terminated":[],
        "truncated":[],
    }

    state_num = env.observation_space.n
    action_num = env.action_space.n

    q_table = np.zeros(shape=(state_num,action_num), dtype=float)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    for episode in range(num_episodes):
        epsilon = linear_epsilon(
            episode,
            epsilon_start,
            epsilon_end,
            epsilon_decay_episodes,
        )
        total_reward, steps, terminated, truncated = train_one_episode(
            env,
            q_table,
            epsilon,
            alpha,
            gamma,
            rng,
            max_steps,
        )

        history["epsilon"].append(epsilon)
        history["rewards"].append(total_reward)
        history["steps"].append(steps)
        history["terminated"].append(terminated)
        history["truncated"].append(truncated)

    return q_table, history



def train_one_episode_sarsa(
        env,
        q_table,
        epsilon,
        alpha,
        gamma,
        rng,
        max_steps,
):
    state, _ = env.reset()
    total_reward = 0
    terminated, truncated = False, False

    action = agent.select_action(q_table, state, epsilon, rng)

    for steps in range(1, max_steps+1):

        next_state, reward, terminated, truncated, info = env.step(action)

        if terminated:
            next_action = None
        else:
            next_action = agent.select_action(q_table, next_state, epsilon, rng)
        
        td_error = agent.update_sarsa(q_table, state, action, reward, next_state, next_action, terminated, alpha, gamma)
        
        total_reward += reward

        if terminated or truncated:
            break

        state = next_state
        action = next_action

    return total_reward, steps, terminated, truncated

def train_sarsa(
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
    history = {
        "epsilon":[],
        "rewards":[],
        "steps":[],
        "terminated":[],
        "truncated":[],
    }

    state_num = env.observation_space.n
    action_num = env.action_space.n

    q_table = np.zeros(shape=(state_num,action_num), dtype=float)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    for episode in range(num_episodes):
        epsilon = linear_epsilon(
            episode,
            epsilon_start,
            epsilon_end,
            epsilon_decay_episodes,
        )
        
        total_reward, steps, terminated, truncated = train_one_episode_sarsa(
            env,
            q_table,
            epsilon,
            alpha,
            gamma,
            rng,
            max_steps,
        )

        history["epsilon"].append(epsilon)
        history["rewards"].append(total_reward)
        history["steps"].append(steps)
        history["terminated"].append(terminated)
        history["truncated"].append(truncated)

    return q_table, history

def linear_epsilon(
        episode,
        epsilon_start,
        epsilon_end,
        decay_episodes,
):
    if episode == 0:
        epsilon = epsilon_start
    elif episode >= decay_episodes:
        epsilon = epsilon_end
    else :
        epsilon = epsilon_start - episode*(epsilon_start - epsilon_end)/decay_episodes

    return epsilon