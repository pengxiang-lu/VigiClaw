import gymnasium as gym
import torch

import mani_skill.envs
import time

# 直接运行环境
env = gym.make(
    "CleanSurfacePlacement",
    num_envs=1,
    obs_mode="rgbd",
    control_mode="pd_joint_delta_pos",  # 或 "pd_joint_pos"
    render_mode="human",
    robot_uids=("piper_x","piper_x"),
)

# 重置环境
obs, info = env.reset(seed=0)
print("环境重置成功")
print(obs["extra"]["tcp_pose"])

# 测试一些动作
while True:
    # 创建字典动作
    if True:
        # action = {
        #     "piper_x-0": [0.415, 0, 0.1, 0, 3.14, 0, 1],
        #     "piper_x-1": [0.415, 0, 0.1, 0, 3.14, 0, 1]
        # }
        action = {
            "piper_x-0": [0,0,0,0,0,0,0],
            "piper_x-1": [0,0,0,0,0,0,0]    }
    else :
        action = torch.zeros(env.action_space.shape[0])
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.1)

env.close()
print("测试完成")