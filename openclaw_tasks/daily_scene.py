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

@register_env("OpenclawDailyScene", max_episode_steps=250)
class DailySceneEnv(BaseEnv):

    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100, WidowXAI]
    obj_spawn_half_size = 0.1
    obj_spawn_center = (0, 0)

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):

        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in OPENCLAW_TASK_CONFIGS:
            cfg = OPENCLAW_TASK_CONFIGS[robot_uids]
        else:
            cfg = OPENCLAW_TASK_CONFIGS["panda"]
        self.goal_thresh = cfg["goal_thresh"]
        self.obj_spawn_half_size = cfg.get("obj_spawn_half_size", 0.1)
        self.obj_spawn_center = cfg.get("obj_spawn_center", (0, 0))
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]

        self._is_first_episode = True
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

        # 更改的部分
        builder = actors.get_actor_builder(
            self.scene,
            id="ycb:055_baseball",
        )
        # 根据物体大小设置合理的初始高度
        builder.initial_pose = sapien.Pose(p=[0.1, -0.1, 0.05])
        self.baseball = builder.build(name="baseball")

        builder = actors.get_actor_builder(
            self.scene,
            id="ycb:011_banana",
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.2, 0.05])
        self.banana = builder.build(name="banana")

        builder = actors.get_actor_builder(
            self.scene,
            id="ycb:065-c_cups",
        )
        builder.initial_pose = sapien.Pose(p=[0.1, 0.1, 0.05])
        self.cup = builder.build(name="cup")

        builder = actors.get_actor_builder(
            self.scene,
            id="ycb:013_apple",
        )
        builder.initial_pose = sapien.Pose(p=[-0.1, 0, 0.05])
        self.apple = builder.build(name="apple")

        builder = actors.get_actor_builder(
            self.scene,
            id="ycb:040_large_marker",
        )
        builder.initial_pose = sapien.Pose(p=[-0.1, 0, 0.05])
        self.marker_pen = builder.build(name="marker pen")

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
            xyz[:, 2] = 0.05
            xy = xyz[:, :2]
            region = [[-0.15, -0.3], [0.15, 0.1]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = 0.08
            baseball_xy = xy + sampler.sample(radius, 100)
            banana_xy = xy + sampler.sample(radius, 100, verbose=False)
            cup_xy = xy + sampler.sample(radius, 100, verbose=False)
            apple_xy = xy + sampler.sample(radius, 100, verbose=False)
            marker_pen_xy = xy + sampler.sample(radius, 100, verbose=False)

            # Baseball
            xyz[:, :2] = baseball_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.baseball.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Banana
            xyz[:, :2] = banana_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.banana.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Cup
            xyz[:, :2] = cup_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cup.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Duplo
            xyz[:, :2] = apple_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.apple.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Marker pen
            xyz[:, :2] = marker_pen_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.marker_pen.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # Skip target selection on first initialization (called by gym.make())
            if not self._is_first_episode:
                self.instruction = "我想要画画，帮我抓取工具"
                print(self.instruction)
            else:
                pass
            self._is_first_episode = False

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                baseball_pose=self.baseball.pose.raw_pose,
                banana_pose=self.banana.pose.raw_pose,
                cup_pose=self.cup.pose.raw_pose,
                apple_pose=self.apple.pose.raw_pose,
                tcp_to_baseball_pos=self.baseball.pose.p - self.agent.tcp.pose.p,
                tcp_to_banana_pos=self.banana.pose.p - self.agent.tcp.pose.p,
                tcp_to_cup_pos=self.cup.pose.p - self.agent.tcp.pose.p,
                tcp_to_apple_pos=self.apple.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def evaluate(self):
        is_robot_static = self.agent.is_static(0.2)

        return {
            "success": torch.zeros(len(is_robot_static), dtype=torch.bool, device=self.device),
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
