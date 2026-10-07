from typing import Any, Union

import numpy as np
import sapien
import torch
from .openclaw_task_cfgs import OPENCLAW_TASK_CONFIGS
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda, WidowXAI, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

@register_env("OpenclawStackCubes", max_episode_steps=250)
class StackCubesEnv(BaseEnv):
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100, WidowXAI]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):

        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in OPENCLAW_TASK_CONFIGS:
            cfg = OPENCLAW_TASK_CONFIGS[robot_uids]
        else:
            cfg = OPENCLAW_TASK_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]

        self._is_first_episode = True
        self._is_first_print = True
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 512, 512, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.red_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="red cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        self.green_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0, 1, 0, 1],
            name="green cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        self.blue_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0, 0, 1, 1],
            name="blue cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        self.yellow_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 1, 0, 1],
            name="yellow cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # Define target TCP pose in world coordinates using Euler angles (XYZ)
            tcp_target_pos = torch.tensor([0.315, 0.0, 0.3], dtype=torch.float32, device=self.device)
            tcp_target_euler = torch.tensor([np.pi, 0, 0], dtype=torch.float32, device=self.device)

            tcp_target_pos = tcp_target_pos.unsqueeze(0).repeat(b, 1)
            tcp_target_euler = tcp_target_euler.unsqueeze(0).repeat(b, 1)

            # Convert Euler angles to quaternion
            tcp_target_rot_matrix = euler_angles_to_matrix(tcp_target_euler, "XYZ")
            tcp_target_quat = matrix_to_quaternion(tcp_target_rot_matrix)

            # Create Pose object for IK
            tcp_target_pose = Pose.create_from_pq(tcp_target_pos, tcp_target_quat)

            # Get current joint positions
            current_qpos = self.agent.robot.get_qpos()

            # Use IK to compute joint positions for arm only
            arm_controller = self.agent.controller.controllers["arm"]
            ik_result = arm_controller.kinematics.compute_ik(
                pose=tcp_target_pose,
                q0=current_qpos,
                is_delta_pose=False,
                solver_config=dict(type="levenberg_marquardt", alpha=1.0)
            )

            # Map IK result to all joints
            # IK returns only controlled joints, need to map to all joints
            init_qpos = current_qpos.clone()
            controlled_joint_indices = arm_controller.active_joint_indices
            init_qpos[:, controlled_joint_indices] = ik_result
            self.agent.robot.set_qpos(init_qpos)

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = self.cube_half_size
            xy = xyz[:, :2]
            region = [[-0.3, -0.2], [0, 0.2]]
            
            # 确保四个方块之间的最小距离不小于5cm (0.05米)
            min_distance = 0.08
            
            # 使用更简单的方法：生成随机位置并检查距离
            radius = torch.linalg.norm(torch.tensor([self.cube_half_size, self.cube_half_size]))
            
            # 生成红色方块位置
            red_cube_xy = torch.rand((b, 2), device=self.device)
            red_cube_xy[:, 0] = red_cube_xy[:, 0] * (region[1][0] - region[0][0]) + region[0][0]
            red_cube_xy[:, 1] = red_cube_xy[:, 1] * (region[1][1] - region[0][1]) + region[0][1]
            
            # 生成绿色方块位置，确保与红色方块距离不小于5cm
            green_cube_xy = torch.zeros_like(red_cube_xy)
            for i in range(b):
                attempts = 0
                while attempts < 100:
                    candidate = torch.rand(2, device=self.device)
                    candidate[0] = candidate[0] * (region[1][0] - region[0][0]) + region[0][0]
                    candidate[1] = candidate[1] * (region[1][1] - region[0][1]) + region[0][1]
                    dist = torch.linalg.norm(red_cube_xy[i] - candidate)
                    if dist >= min_distance:
                        green_cube_xy[i] = candidate
                        break
                    attempts += 1
                if attempts >= 100:
                    # 如果找不到合适位置，使用默认位置（与红色方块有一定偏移）
                    green_cube_xy[i] = red_cube_xy[i] + torch.tensor([min_distance, 0], device=self.device)
            
            # 生成蓝色方块位置，确保与红色和绿色方块距离都不小于5cm
            blue_cube_xy = torch.zeros_like(red_cube_xy)
            for i in range(b):
                attempts = 0
                while attempts < 100:
                    candidate = torch.rand(2, device=self.device)
                    candidate[0] = candidate[0] * (region[1][0] - region[0][0]) + region[0][0]
                    candidate[1] = candidate[1] * (region[1][1] - region[0][1]) + region[0][1]
                    dist_to_red = torch.linalg.norm(red_cube_xy[i] - candidate)
                    dist_to_green = torch.linalg.norm(green_cube_xy[i] - candidate)
                    if dist_to_red >= min_distance and dist_to_green >= min_distance:
                        blue_cube_xy[i] = candidate
                        break
                    attempts += 1
                if attempts >= 100:
                    # 如果找不到合适位置，使用默认位置
                    blue_cube_xy[i] = red_cube_xy[i] + torch.tensor([0, min_distance], device=self.device)
            
            # 生成黄色方块位置，确保与红色、绿色和蓝色方块距离都不小于5cm
            yellow_cube_xy = torch.zeros_like(red_cube_xy)
            for i in range(b):
                attempts = 0
                while attempts < 100:
                    candidate = torch.rand(2, device=self.device)
                    candidate[0] = candidate[0] * (region[1][0] - region[0][0]) + region[0][0]
                    candidate[1] = candidate[1] * (region[1][1] - region[0][1]) + region[0][1]
                    dist_to_red = torch.linalg.norm(red_cube_xy[i] - candidate)
                    dist_to_green = torch.linalg.norm(green_cube_xy[i] - candidate)
                    dist_to_blue = torch.linalg.norm(blue_cube_xy[i] - candidate)
                    if (dist_to_red >= min_distance and 
                        dist_to_green >= min_distance and 
                        dist_to_blue >= min_distance):
                        yellow_cube_xy[i] = candidate
                        break
                    attempts += 1
                if attempts >= 100:
                    # 如果找不到合适位置，使用默认位置
                    yellow_cube_xy[i] = red_cube_xy[i] + torch.tensor([-min_distance, 0], device=self.device)

            # Red cube
            xyz[:, :2] = red_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.red_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Green cube
            xyz[:, :2] = green_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.green_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Blue cube
            xyz[:, :2] = blue_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.blue_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Yellow cube
            xyz[:, :2] = yellow_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.yellow_cube.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # Skip target selection on first initialization (called by gym.make())
            if not self._is_first_episode:
                self.instruction = "将红色方块堆叠在绿色方块上，再将蓝色方块堆叠在红色方块上，最后将黄色方块堆叠在蓝色方块上"
                print(self.instruction)
            else:
                self._is_first_episode = False



    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                red_cube_pose=self.red_cube.pose.raw_pose,
                green_cube_pose=self.green_cube.pose.raw_pose,
                blue_cube_pose=self.blue_cube.pose.raw_pose,
                yellow_cube_pose=self.yellow_cube.pose.raw_pose,
                tcp_to_red_cube_pos=self.red_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_green_cube_pos=self.green_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_blue_cube_pos=self.blue_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_yellow_cube_pos=self.yellow_cube.pose.p - self.agent.tcp.pose.p,
                red_cube_to_green_cube_pos=self.green_cube.pose.p - self.red_cube.pose.p,
                green_cube_to_blue_cube_pos=self.blue_cube.pose.p - self.green_cube.pose.p,
                blue_cube_to_yellow_cube_pos=self.yellow_cube.pose.p - self.blue_cube.pose.p,
                red_cube_to_blue_cube_pos=self.blue_cube.pose.p - self.red_cube.pose.p,
                red_cube_to_yellow_cube_pos=self.yellow_cube.pose.p - self.red_cube.pose.p,
                green_cube_to_yellow_cube_pos=self.yellow_cube.pose.p - self.green_cube.pose.p,
            )
        return obs

    def evaluate(self):
        pos_red = self.red_cube.pose.p
        pos_green = self.green_cube.pose.p
        pos_blue = self.blue_cube.pose.p
        pos_yellow = self.yellow_cube.pose.p

        offset_red_green = pos_red - pos_green
        offset_blue_green = pos_blue - pos_green
        offset_blue_red = pos_blue - pos_red
        offset_yellow_green = pos_yellow - pos_green
        offset_yellow_red = pos_yellow - pos_red
        offset_yellow_blue = pos_yellow - pos_blue

        def evaluate_cube_distance(offset, cube_a, cube_b, top_or_next):
            xy_flag = (
                torch.linalg.norm(offset[..., :2], axis=1)
                <= 2 * self.cube_half_size + 0.005
            )
            z_flag = torch.abs(offset[..., 2]) > 2 * self.cube_half_size
            if top_or_next == "top":
                is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
            elif top_or_next == "next_to":
                is_cubeA_on_cubeB = xy_flag
            else:
                return NotImplementedError(
                    f"Expect top_or_next to be either 'top' or 'next_to', got {top_or_next}"
                )

            is_cubeA_static = cube_a.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            is_cubeA_grasped = self.agent.is_grasping(cube_a)

            success = is_cubeA_on_cubeB & is_cubeA_static & (~is_cubeA_grasped)
            return success.bool()

        success_red_on_green = evaluate_cube_distance(
            offset_red_green, self.red_cube, self.green_cube, "top"
        )
        success_blue_on_red = evaluate_cube_distance(
            offset_blue_red, self.blue_cube, self.red_cube, "top"
        )
        success_yellow_on_blue = evaluate_cube_distance(
            offset_yellow_blue, self.yellow_cube, self.blue_cube, "top"
        )
        
        # 四个方块的堆叠顺序：绿色(底) -> 红色 -> 蓝色 -> 黄色(顶)
        success = torch.logical_and(
            success_red_on_green, 
            torch.logical_and(
                success_blue_on_red,
                success_yellow_on_blue
            )
        )
        is_robot_static = self.agent.is_static(0.2)

        return {
            "success": success,
            "is_robot_static": is_robot_static,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):

        return 0

    def compute_normalized_dense_reward(
            self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    def get_instruction(self):
        return self.instruction
