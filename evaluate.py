import numpy as np
import agent

def evaluate_policy(
        env,
        q_table,
        num_episodes,
        max_steps,
        seed,
):
    epsilon = 0.0
    evaluate_history = {
        "rewards":[],
        "steps":[],
        "terminated":[],
        "truncated":[],
    }

    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    for _ in range(num_episodes):
        state, _ = env.reset()
        total_reward, steps = 0.0, 0
        terminated, truncated = False, False

        while steps < max_steps :
            action = agent.select_action(q_table, state, epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break

        evaluate_history["rewards"].append(total_reward)
        evaluate_history["steps"].append(steps)
        evaluate_history["terminated"].append(terminated)
        evaluate_history["truncated"].append(truncated)

    return evaluate_history

