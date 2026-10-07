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

# 任务描述：将香蕉放在盘子上
# 任务完成判断条件：香蕉的中心只要在盘子上就可以

@register_env("OpenclawPutBananaOnPlate", max_episode_steps=50)
class PutBananaOnPlateEnv(BaseEnv):
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
        self.goal_thresh = cfg["goal_thresh"]
        self.banana_half_size = 0.2
        self.banana_center = [-0.1,0,0.03]
        self.plate_half_size = 0.2
        self.plate_center = [-0.1,0,0.03]

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

        builder = actors.get_actor_builder(
            self.scene,
            id="ycb:011_banana",
        )
        builder.initial_pose = sapien.Pose(p=[0, 0.2, 0.05])
        self.banana = builder.build(name="banana")

        builder = actors.get_actor_builder(
            self.scene,
            id="ycb:029_plate",
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.2, 0.05])
        self.plate = builder.build(name="plate")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Reset robot to initial pose using IK
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

            center_x = -0.1
            center_y = 0
            center_z = 0.03
            min_distance = 0.15
            max_distance = 0.2

            banana_angles = torch.rand(b) * 2 * np.pi
            plate_angles = (banana_angles + np.pi) % (2 * np.pi)

            banana_distances = min_distance + torch.rand(b) * (max_distance - min_distance)
            plate_distances = min_distance + torch.rand(b) * (max_distance - min_distance)

            banana_xyz = torch.zeros((b, 3))
            banana_xyz[:, 0] = center_x + banana_distances * torch.cos(banana_angles)
            banana_xyz[:, 1] = center_y + banana_distances * torch.sin(banana_angles)
            banana_xyz[:, 2] = center_z

            banana_qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.banana.set_pose(Pose.create_from_pq(banana_xyz, banana_qs))

            plate_xyz = torch.zeros((b, 3))
            plate_xyz[:, 0] = center_x + plate_distances * torch.cos(plate_angles)
            plate_xyz[:, 1] = center_y + plate_distances * torch.sin(plate_angles)
            plate_xyz[:, 2] = center_z

            plate_qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.plate.set_pose(Pose.create_from_pq(plate_xyz, plate_qs))

            # Skip target selection on first initialization (called by gym.make())
            if not self._is_first_episode:
                self.target_object = self.banana
                self.instruction = f"把{self.target_object.name}放在盘子上"
                print(self.instruction)
            else:
                self.target_object = self.banana
            self._is_first_episode = False

    def _get_obs_extra(self, info: dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.target_object.pose.raw_pose,
                tcp_to_obj_pos=self.target_object.pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
                torch.linalg.norm(self.target_object.pose.p[:, 0:2] - self.plate.pose.p[:, 0:2], axis=1)
                <= 0.1
        ) & (
                self.target_object.pose.p[:, 2] < 0.1
        )
        is_grasped = self.agent.is_grasping(self.target_object)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static & (~is_grasped),
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):

        return 0

    def compute_normalized_dense_reward(
            self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    def get_instruction(self):
        return self.instruction
