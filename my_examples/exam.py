from pathlib import Path
import os

# Use the project-level asset directory regardless of the current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MS_ASSET_DIR", str(PROJECT_ROOT / "assets"))

# 创建 test_run.py
import gymnasium as gym
import mani_skill.envs
import time
import numpy as np
from mani_skill.examples.motionplanning.double_piper_x.solutions import solveMyPickCube


def test_bimanual_environment():
    """测试双臂环境"""
    env = gym.make(
        "MyPickCube-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_pos",  # 使用关节位置控制
        render_mode="human",  # 使用GUI查看
        robot_uids=("piper_x", "piper_x")
    )

    print(f"环境创建成功")
    print(f"动作空间: {env.action_space}")
    print(f"观察空间: {env.observation_space}")

    # 测试重置
    obs, info = env.reset(seed=0)
    print(f"环境重置成功，观察形状: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")

    # 检查是否为双臂
    env_unwrapped = env.unwrapped
    print(f"是否为双臂系统: {getattr(env_unwrapped, 'is_bimanual', 'Unknown')}")
    print(f"机器人配置: {getattr(env_unwrapped, 'robot_uids', 'Unknown')}")

    # 运行解决方案
    print("\n=== 开始运动规划 ===")
    result = solveMyPickCube(env, seed=0, debug=True, vis=True)

    print(f"\n=== 结果 ===")
    print(f"成功: {result.get('success', False)}")
    print(f"步数: {result.get('elapsed_steps', 0)}")
    print(f"抓取状态: {result.get('is_grasped', False)}")
    print(f"物体放置: {result.get('is_obj_placed', False)}")
    print(f"错误: {result.get('error', 'None')}")

    env.close()

    return result


if __name__ == "__main__":
    test_bimanual_environment()
