import gymnasium as gym
import numpy as np
import evaluate
import config
import artifacts

def main():

    q_table = artifacts.load_q_table(config.Q_TABLE_PATH)
    evaluate_env = gym.make(
                id = config.ENV_ID,
                map_name = config.MAP,
                is_slippery = config.IS_SLIDE,
            )
    try:
        state_n = evaluate_env.observation_space.n
        action_n = evaluate_env.action_space.n
        wish_shape = (state_n, action_n)

        if q_table.shape != wish_shape:
            raise ValueError(q_table.shape, wish_shape)

        evaluate_history = evaluate.evaluate_policy(evaluate_env, q_table, config.EVALUATE_TIMES, config.MAX_STEP, config.EVALUATE_SEED)

    finally:
        evaluate_env.close()

    print("q_table_loaded_from:", config.Q_TABLE_PATH)
    print("evaluate_success_rate:", np.mean(evaluate_history["rewards"]))
    print("evaluate_average_steps:", np.mean(evaluate_history["steps"]))

if __name__ == "__main__":
    main()