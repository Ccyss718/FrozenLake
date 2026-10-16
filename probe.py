"""FrozenLake 环境接口探针。

"""

import gymnasium as gym


ENV_ID = "FrozenLake-v1"
MAP_NAME = "4x4"
SEED = 42
MAX_STEPS = 100


def main():
    env = gym.make(
        ENV_ID,
        map_name=MAP_NAME,
        is_slippery=False,
        render_mode="ansi",
    )

    try:
        # TODO 1：打印 observation_space、action_space 以及它们的 n。
        print("observation_space:", env.observation_space)
        print("observation_cnt:", env.observation_space.n)
        print("action_space:", env.action_space)
        print("action_cnt:", env.action_space.n)

        # TODO 2：调用 reset(seed=SEED)，接收 observation 和 info 并打印。
        state, info = env.reset(seed = SEED)
        print("state, info:", state, info)

        # TODO 3：使用 env.action_space.sample() 随机选择动作。
        action = env.action_space.sample()

        # TODO 4：调用一次 step(action)，接收并打印它返回的五个值。
        next_state, reward, terminated, truncated, info_a = env.step(action)
        print(next_state, reward, terminated, truncated, info_a)

        # TODO 5：重新 reset，然后运行一个完整的随机策略回合：
        # - 每一步重新随机采样动作；
        # - 累加回报并记录步数；
        # - terminated 或 truncated 为 True 时立即结束；
        # - 最多执行 MAX_STEPS 步，避免循环失控；
        # - 最后打印总回报、总步数以及结束原因。
        state, info = env.reset(seed = SEED)
        total_reward = 0
        steps = 0
        end_reason = "max_steps"
        while(steps<MAX_STEPS):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info_a = env.step(action)
            total_reward += reward
            steps +=1
            state = next_state

            if terminated:
                end_reason = "terminated"
                break

            if truncated:
                end_reason = "truncated"
                break

        print("end_reason:", end_reason)
        print("total_reward:", total_reward)
        print("total_steps:", steps)
        
    finally:
        env.close()


if __name__ == "__main__":
    main()
