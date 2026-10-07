from typing import Any, Union
import numpy as np
import sapien
import torch
from mani_skill.envs.utils import randomization
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.registration import register_env
from .stack_cubes import StackCubesEnv


@register_env("OpenclawStackCubesDisturb", max_episode_steps=250)
class StackCubesDisturbEnv(StackCubesEnv):
    """堆叠方块任务，带有干扰功能：当红色方块放到绿色方块上后2-3秒，红色方块会瞬移回桌面"""
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
        
        # 干扰相关状态
        self.disturbance_triggered = False
        self.stack_success_time = None
        self.disturbance_type = None  # 'red' 或 'blue'
        
        self.disturbance_delay = 2.0  # 固定延迟2秒
        self.current_time = 0.0
        self.dt = 0.02  # 时间步长
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        
        # 随机选择干扰类型：'red' 或 'blue'
        self.disturbance_type = np.random.choice(['red', 'blue'])
        
        # 重置干扰状态
        self.disturbance_triggered = False
        self.stack_success_time = None
        
        self.disturbance_delay = 2.0
        self.current_time = 0.0
        
        # 只在实际的reset时打印调试信息（跳过第一次初始化）
        print(f"[DEBUG] Episode initialized: 干扰类型={self.disturbance_type}, current_time={self.current_time}")


    def check_red_on_green(self):
        """检查红色方块是否在绿色方块上"""
        pos_red = self.red_cube.pose.p
        pos_green = self.green_cube.pose.p
        
        offset = pos_red - pos_green
        
        # 检查水平距离
        xy_dist = torch.linalg.norm(offset[..., :2], axis=1)
        xy_flag = xy_dist <= 2 * self.cube_half_size + 0.005
        
        # 检查垂直距离（红色方块应该在绿色方块上方）
        z_flag = offset[..., 2] > 2 * self.cube_half_size - 0.01
        z_flag = z_flag & (offset[..., 2] < 2 * self.cube_half_size + 0.05)

        red_on_green = xy_flag & z_flag
        
        return red_on_green.bool()
    
    def check_blue_on_red(self):
        """检查蓝色方块是否在红色方块上"""
        pos_blue = self.blue_cube.pose.p
        pos_red = self.red_cube.pose.p
        
        offset = pos_blue - pos_red
        
        # 检查水平距离
        xy_dist = torch.linalg.norm(offset[..., :2], axis=1)
        xy_flag = xy_dist <= 2 * self.cube_half_size + 0.005
        
        # 检查垂直距离（蓝色方块应该在红色方块上方）
        z_flag = offset[..., 2] > 2 * self.cube_half_size - 0.01
        z_flag = z_flag & (offset[..., 2] < 2 * self.cube_half_size + 0.05)

        blue_on_red = xy_flag & z_flag
        
        return blue_on_red.bool()
    
    def check_gripper_released(self, cube):
        """检查机械臂夹爪是否松开指定方块"""
        is_cube_grasped = self.agent.is_grasping(cube)
        return (~is_cube_grasped).bool()
    
    def apply_disturbance(self):
        """应用干扰：将指定方块瞬移回桌面的随机位置，确保距离其他方块不少于5cm"""
        b = self.num_envs
        
        # 在桌面范围内随机选择位置
        region = [[-0.3, -0.2], [0, 0.2]]
        sampler = randomization.UniformPlacementSampler(
            bounds=region, batch_size=b, device=self.device
        )
        radius = torch.linalg.norm(torch.tensor([self.cube_half_size, self.cube_half_size]))
        
        # 生成新的随机位置，并检查距离其他方块是否足够
        max_attempts = 100
        for attempt in range(max_attempts):
            xy = sampler.sample(radius, 100)
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, :2] = xy
            xyz[:, 2] = self.cube_half_size  # 放在桌面上
            
            # 检查新位置是否距离其他方块至少5cm
            valid_position = True
            
            # 获取所有其他方块的位置
            other_cubes = []
            if self.disturbance_type == 'red':
                other_cubes = [self.green_cube, self.blue_cube, self.yellow_cube]
            elif self.disturbance_type == 'blue':
                other_cubes = [self.red_cube, self.green_cube, self.yellow_cube]
            
            for cube in other_cubes:
                cube_pos = cube.pose.p
                # 计算水平距离
                xy_dist = torch.linalg.norm(xyz[:, :2] - cube_pos[:, :2], axis=1)
                # 检查是否至少5cm距离
                if (xy_dist < 0.05).any():
                    valid_position = False
                    break
            
            if valid_position:
                break
            
            if attempt == max_attempts - 1:
                print(f"[WARNING] 无法找到满足5cm距离要求的位置，使用最后一次尝试的位置")
        
        # 随机旋转
        qs = randomization.random_quaternions(
            b,
            lock_x=True,
            lock_y=True,
            lock_z=False,
        )
        
        # 根据干扰类型选择要瞬移的方块
        if self.disturbance_type == 'red':
            self.red_cube.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            print(f"[红色方块干扰触发！红色方块已瞬移回桌面（延迟2秒），距离其他方块至少5cm")
        elif self.disturbance_type == 'blue':
            self.blue_cube.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            print(f"[蓝色方块干扰触发！蓝色方块已瞬移回桌面（延迟2秒），距离其他方块至少5cm")
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 更新时间
        self.current_time += self.dt
        
        # 检查是否需要触发干扰
        if not self.disturbance_triggered:
            
            # 根据干扰类型检查相应的条件
            if self.disturbance_type == 'red':
                cube_on_target = self.check_red_on_green()
                target_cube = self.red_cube
            elif self.disturbance_type == 'blue':
                cube_on_target = self.check_blue_on_red()
                target_cube = self.blue_cube
            else:
                return obs, reward, terminated, truncated, info
            
            gripper_released = self.check_gripper_released(target_cube)
            
            # 只有当方块在目标位置上且夹爪松开时才开始计时
            if cube_on_target.any() and gripper_released.any():
                if self.stack_success_time is None:
                    self.stack_success_time = self.current_time
                    cube_name = "红色方块" if self.disturbance_type == 'red' else "蓝色方块"
                    print(f"[DEBUG] 检测到{cube_name}在目标位置上且夹爪已松开，将在2秒后触发干扰 (current_time={self.current_time:.2f})")
            else:
                # 如果方块不在目标位置上或夹爪又抓取了方块，重置计时
                if self.stack_success_time is not None:
                    print(f"[DEBUG] 条件不再满足，重置计时 (cube_on_target={cube_on_target.any()}, gripper_released={gripper_released.any()})")
                self.stack_success_time = None
            
            # 检查是否达到延迟时间（固定2秒）
            if self.stack_success_time is not None:
                time_since_stack = self.current_time - self.stack_success_time
                if time_since_stack >= 2.0:
                    print(f"[DEBUG] 达到延迟时间，触发干扰 (time_since_stack={time_since_stack:.2f})")
                    self.apply_disturbance()
                    self.disturbance_triggered = True
        
        return obs, reward, terminated, truncated, info
    
    def get_instruction(self):
        base_instruction = super().get_instruction()
        if base_instruction:
            return base_instruction
        return "将红色方块堆叠在绿色方块上，再将蓝色方块堆叠在红色方块上 +（注意：物块可能会受到干扰！）"