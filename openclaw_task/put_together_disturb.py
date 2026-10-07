from typing import Any, Union
import numpy as np
import sapien
import torch
from mani_skill.envs.utils import randomization
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.registration import register_env
from .put_together import PutTogetherEnv


@register_env("OpenclawPutTogetherDisturb", max_episode_steps=250)
class PutTogetherDisturbEnv(PutTogetherEnv):
    """将方块放进对应颜色的碗任务，带有干扰功能：
    1. 当物块放入对应的碗中时，将放在碗里的物块移动回原来的位置
    2. 第一个物块放入对应的碗里后，随机将剩下两个碗中的一个碗向前移动一段距离(0.1m)
    这两种干扰随机出现，但只会出现一种干扰
    """

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

        # 干扰相关状态
        self.disturbance_triggered = False
        self.first_cube_in_bowl_time = None
        self.disturbance_type = None  # 'cube_reset' 或 'bowl_move'
        self.cube_reset_trigger = None  # cube_reset 干扰触发时机：1 或 2
        self.cube_initial_poses = {}  # 记录每个物块的初始位置
        self.bowl_initial_poses = {}  # 记录每个碗的初始位置
        self.previous_cubes_in_bowls = set()  # 记录上一帧在碗中的物块颜色

        self.disturbance_delay = 2.0  # 固定延迟2秒
        self.current_time = 0.0
        self.dt = 0.02  # 时间步长

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        # 随机选择干扰类型：'cube_reset' 或 'bowl_move'
        self.disturbance_type = np.random.choice(['cube_reset', 'bowl_move'])

        # 如果是 cube_reset 干扰类型，随机选择触发时机（1 或 2）
        if self.disturbance_type == 'cube_reset':
            self.cube_reset_trigger = np.random.choice([1, 2])
        else:
            self.cube_reset_trigger = None

        # 重置干扰状态
        self.disturbance_triggered = False
        self.first_cube_in_bowl_time = None
        self.previous_cubes_in_bowls = set()
        self.newly_placed_cube_color = None  # 记录刚刚放进去的物块颜色

        self.disturbance_delay = 2.0
        self.current_time = 0.0

        # 记录物块和碗的初始位置
        self.cube_initial_poses = {
            'red': self.red_cube.pose.p.clone(),
            'green': self.green_cube.pose.p.clone(),
            'blue': self.blue_cube.pose.p.clone()
        }
        self.bowl_initial_poses = {
            'red': self.red_bowl.pose.p.clone(),
            'green': self.green_bowl.pose.p.clone(),
            'blue': self.blue_bowl.pose.p.clone()
        }

        # 只在实际的reset时打印调试信息（跳过第一次初始化）
        if not self._is_first_episode:
            trigger_info = f", 触发时机=第{self.cube_reset_trigger}个物块" if self.disturbance_type == 'cube_reset' else ""
            print(f"[DEBUG] Episode initialized: 干扰类型={self.disturbance_type}{trigger_info}, current_time={self.current_time}")
        self._is_first_episode = False

    def check_cube_in_bowl(self, cube, bowl):
        """检查物块是否在碗中"""
        cube_pos = cube.pose.p
        bowl_pos = bowl.pose.p

        offset = cube_pos - bowl_pos

        # 检查水平距离
        xy_dist = torch.linalg.norm(offset[..., :2], axis=1)
        xy_flag = xy_dist <= 0.08

        # 检查垂直距离（物块应该在碗内）
        z_flag = torch.abs(offset[..., 2]) <= 0.05

        # 检查物块是否静止
        is_cube_static = cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)

        cube_in_bowl = xy_flag & z_flag & is_cube_static

        return cube_in_bowl.bool()

    def check_gripper_released(self, cube):
        """检查机械臂夹爪是否松开指定物块"""
        is_cube_grasped = self.agent.is_grasping(cube)
        return (~is_cube_grasped).bool()

    def get_cube_bowl_pairs(self):
        """获取物块和对应的碗"""
        return [
            ('red', self.red_cube, self.red_bowl),
            ('green', self.green_cube, self.green_bowl),
            ('blue', self.blue_cube, self.blue_bowl)
        ]

    def check_any_cube_in_correct_bowl(self):
        """检查是否有物块在对应的碗中"""
        cube_bowl_pairs = self.get_cube_bowl_pairs()

        for color, cube, bowl in cube_bowl_pairs:
            cube_in_bowl = self.check_cube_in_bowl(cube, bowl)
            gripper_released = self.check_gripper_released(cube)

            if cube_in_bowl.any() and gripper_released.any():
                return color, cube, bowl

        return None, None, None
    
    def get_all_cubes_in_correct_bowls(self):
        """获取所有在对应碗中的物块列表"""
        cubes_in_bowls = []
        cube_bowl_pairs = self.get_cube_bowl_pairs()

        for color, cube, bowl in cube_bowl_pairs:
            cube_in_bowl = self.check_cube_in_bowl(cube, bowl)
            gripper_released = self.check_gripper_released(cube)

            if cube_in_bowl.any() and gripper_released.any():
                cubes_in_bowls.append((color, cube, bowl))

        return cubes_in_bowls

    def apply_cube_reset_disturbance(self):
        """应用干扰：将刚刚放进去的那个物块移动回原来的位置"""
        if self.newly_placed_cube_color is None:
            print(f"[WARNING] 无法确定刚刚放进去的物块，跳过干扰")
            return
        
        # 根据颜色获取对应的物块
        if self.newly_placed_cube_color == 'red':
            selected_cube = self.red_cube
        elif self.newly_placed_cube_color == 'green':
            selected_cube = self.green_cube
        elif self.newly_placed_cube_color == 'blue':
            selected_cube = self.blue_cube
        else:
            print(f"[WARNING] 未知的物块颜色：{self.newly_placed_cube_color}，跳过干扰")
            return
        
        b = self.num_envs

        # 获取物块的初始位置
        initial_pos = self.cube_initial_poses[self.newly_placed_cube_color]

        # 随机旋转
        qs = randomization.random_quaternions(
            b,
            lock_x=True,
            lock_y=True,
            lock_z=False,
        )

        # 将物块移动回初始位置
        selected_cube.set_pose(Pose.create_from_pq(p=initial_pos, q=qs))
        print(f"[物块重置干扰触发！{self.newly_placed_cube_color}色物块（刚刚放进去的）已移动回初始位置（延迟2秒）")
        self.newly_placed_cube_color = None

    def apply_bowl_move_disturbance(self, first_color):
        """应用干扰：随机将剩下两个碗中的一个碗向前移动0.1m"""
        b = self.num_envs

        # 获取剩下的两个碗
        remaining_bowls = []
        bowl_colors = ['red', 'green', 'blue']

        for color in bowl_colors:
            if color != first_color:
                if color == 'red':
                    remaining_bowls.append(('red', self.red_bowl))
                elif color == 'green':
                    remaining_bowls.append(('green', self.green_bowl))
                elif color == 'blue':
                    remaining_bowls.append(('blue', self.blue_bowl))

        # 随机选择一个碗
        selected_color, selected_bowl = remaining_bowls[np.random.choice(len(remaining_bowls))]

        # 获取碗的当前位置
        current_pos = selected_bowl.pose.p

        # 向前移动0.1m（沿x轴正方向）
        new_pos = current_pos.clone()
        new_pos[:, 0] += 0.1

        # 保持旋转不变
        current_q = selected_bowl.pose.q

        # 移动碗
        selected_bowl.set_pose(Pose.create_from_pq(p=new_pos, q=current_q))
        print(f"[碗移动干扰触发！{selected_color}色碗已向前移动0.1m（延迟2秒）")

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # 更新时间
        self.current_time += self.dt

        # 检查是否需要触发干扰
        if not self.disturbance_triggered:
            # 获取所有在对应碗中的物块
            cubes_in_bowls = self.get_all_cubes_in_correct_bowls()
            num_cubes_in_bowls = len(cubes_in_bowls)
            
            # 确定刚刚放进去的物块
            current_cubes_in_bowls = set(color for color, cube, bowl in cubes_in_bowls)
            if hasattr(self, 'previous_cubes_in_bowls'):
                new_cubes = current_cubes_in_bowls - self.previous_cubes_in_bowls
                if len(new_cubes) == 1:
                    self.newly_placed_cube_color = list(new_cubes)[0]
                    print(f"[DEBUG] 检测到新放入的物块：{self.newly_placed_cube_color}")
            self.previous_cubes_in_bowls = current_cubes_in_bowls

            if self.disturbance_type == 'cube_reset':
                # cube_reset 干扰类型：检查是否达到触发时机
                if num_cubes_in_bowls == self.cube_reset_trigger and self.newly_placed_cube_color is not None:
                    # 检查刚刚放进去的物块是否已松开夹爪
                    if self.newly_placed_cube_color == 'red':
                        cube_released = self.check_gripper_released(self.red_cube)
                    elif self.newly_placed_cube_color == 'green':
                        cube_released = self.check_gripper_released(self.green_cube)
                    elif self.newly_placed_cube_color == 'blue':
                        cube_released = self.check_gripper_released(self.blue_cube)
                    else:
                        cube_released = torch.tensor([False], device=self.device)
                    
                    if cube_released.all():
                        if self.first_cube_in_bowl_time is None:
                            self.first_cube_in_bowl_time = self.current_time
                            print(f"[DEBUG] 检测到第{num_cubes_in_bowls}个物块（{self.newly_placed_cube_color}色）在对应的碗中且夹爪已松开，将在2秒后触发干扰 (current_time={self.current_time:.2f})")
                else:
                    # 如果物块数量不匹配或无法确定新物块，重置计时
                    if self.first_cube_in_bowl_time is not None:
                        print(f"[DEBUG] 条件不再满足，重置计时 (当前数量={num_cubes_in_bowls}, 触发时机={self.cube_reset_trigger}, 新物块={self.newly_placed_cube_color})")
                    self.first_cube_in_bowl_time = None
            elif self.disturbance_type == 'bowl_move':
                # bowl_move 干扰类型：检查是否有物块在对应的碗中
                color, cube, bowl = self.check_any_cube_in_correct_bowl()
                if color is not None and cube is not None and bowl is not None:
                    if self.first_cube_in_bowl_time is None:
                        self.first_cube_in_bowl_time = self.current_time
                        print(f"[DEBUG] 检测到{color}色物块在对应的碗中且夹爪已松开，将在2秒后触发干扰 (current_time={self.current_time:.2f})")
                else:
                    # 如果没有物块在对应的碗中，重置计时
                    if self.first_cube_in_bowl_time is not None:
                        print(f"[DEBUG] 条件不再满足，重置计时")
                    self.first_cube_in_bowl_time = None

            # 检查是否达到延迟时间（固定2秒）
            if self.first_cube_in_bowl_time is not None:
                time_since_in_bowl = self.current_time - self.first_cube_in_bowl_time
                if time_since_in_bowl >= 2.0:
                    print(f"[DEBUG] 达到延迟时间，触发干扰 (time_since_in_bowl={time_since_in_bowl:.2f})")

                    # 根据干扰类型应用相应的干扰
                    if self.disturbance_type == 'cube_reset':
                        # 随机选择一个已经在碗里的物块，将其移动回原来的位置
                        self.apply_cube_reset_disturbance()
                    elif self.disturbance_type == 'bowl_move':
                        # 随机将剩下两个碗中的一个碗向前移动0.1m
                        color, cube, bowl = self.check_any_cube_in_correct_bowl()
                        if color is not None:
                            self.apply_bowl_move_disturbance(color)

                    self.disturbance_triggered = True

        return obs, reward, terminated, truncated, info

    def get_instruction(self):
        base_instruction = super().get_instruction()
        if base_instruction:
            return base_instruction + "（注意：物块或碗可能会受到干扰！）"
        return "将方块放进对应颜色的碗里（注意：物块或碗可能会受到干扰！）"