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

from mani_skill import ASSET_DIR


def get_ycb_builder_with_color(scene, ycb_id, color, scale=1.0):
    """
    获取 YCB 模型的 builder，使用统一颜色替代纹理

    Args:
        scene: 场景对象
        ycb_id: YCB 模型 ID（如 "013_apple"）
        color: 颜色 [R, G, B, A]
        scale: 缩放比例
    """
    builder = actors.get_actor_builder(
        scene,
        id=f"ycb:{ycb_id}",
        add_collision=True,
        add_visual=False,
    )

    model_dir = ASSET_DIR / "assets/mani_skill2_ycb/models" / ycb_id
    visual_file = str(model_dir / "textured.obj")

    builder.add_visual_from_file(
        filename=visual_file,
        scale=[scale] * 3,
        material=sapien.render.RenderMaterial(
            base_color=color,
            roughness=0.5,
            specular=0.0,
        ),
    )

    return builder

@register_env("OpenclawPutTogether", max_episode_steps=250)
class PutTogetherEnv(BaseEnv):
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

        builder = get_ycb_builder_with_color(
            self.scene,
            ycb_id="024_bowl",
            color=[1.0, 0.0, 0.0, 1.0]
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.2, 0.05])
        self.red_bowl = builder.build(name="red bowl")

        builder = get_ycb_builder_with_color(
            self.scene,
            ycb_id="024_bowl",
            color=[0.0, 1.0, 0.0, 1.0]
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.05])
        self.green_bowl = builder.build(name="green bowl")

        builder = get_ycb_builder_with_color(
            self.scene,
            ycb_id="024_bowl",
            color=[0.0, 0.0, 1.0, 1.0]
        )
        builder.initial_pose = sapien.Pose(p=[0, 0.2, 0.05])
        self.blue_bowl = builder.build(name="blue bowl")

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

            bowl_positions = [
                [-0.1, -0.2, 0.05],
                [-0.1, 0, 0.05],
                [-0.1, 0.2, 0.05]
            ]
            bowl_colors = ["red", "green", "blue"]
            bowl_objects = [self.red_bowl, self.green_bowl, self.blue_bowl]
            cube_objects = [self.red_cube, self.green_cube, self.blue_cube]

            bowl_perm = torch.randperm(3)
            cube_perm = torch.randperm(3)

            for i in range(3):
                bowl_idx = bowl_perm[i].item()
                bowl_objects[bowl_idx].set_pose(
                    Pose.create_from_pq(
                        p=torch.tensor(bowl_positions[i], device=self.device).unsqueeze(0).repeat(b, 1),
                        q=torch.tensor([0, 0, 0, 1], device=self.device).unsqueeze(0).repeat(b, 1)
                    )
                )

            for i in range(3):
                cube_idx = cube_perm[i].item()
                bowl_idx = bowl_perm[i].item()
                bowl_pos = torch.tensor(bowl_positions[i], device=self.device)
                cube_pos = bowl_pos.clone()
                distance = 0.15 + torch.rand(1, device=self.device).item() * 0.1
                cube_pos[0] -= distance
                cube_pos[2] = self.cube_half_size

                qs = randomization.random_quaternions(
                    b,
                    lock_x=True,
                    lock_y=True,
                    lock_z=False,
                )
                cube_objects[cube_idx].set_pose(
                    Pose.create_from_pq(
                        p=cube_pos.unsqueeze(0).repeat(b, 1),
                        q=qs
                    )
                )

            if not self._is_first_episode:
                self.instruction = "将方块放进对应颜色的碗里"
                print(self.instruction)
            else:
                pass
            self._is_first_episode = False

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                red_cube_pose=self.red_cube.pose.raw_pose,
                green_cube_pose=self.green_cube.pose.raw_pose,
                blue_cube_pose=self.blue_cube.pose.raw_pose,
                red_bowl_pose=self.red_bowl.pose.raw_pose,
                green_bowl_pose=self.green_bowl.pose.raw_pose,
                blue_bowl_pose=self.blue_bowl.pose.raw_pose,
                tcp_to_red_cube_pos=self.red_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_green_cube_pos=self.green_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_blue_cube_pos=self.blue_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_red_bowl_pos=self.red_bowl.pose.p - self.agent.tcp.pose.p,
                tcp_to_green_bowl_pos=self.green_bowl.pose.p - self.agent.tcp.pose.p,
                tcp_to_blue_bowl_pos=self.blue_bowl.pose.p - self.agent.tcp.pose.p,
                red_cube_to_red_bowl_pos=self.red_bowl.pose.p - self.red_cube.pose.p,
                green_cube_to_green_bowl_pos=self.green_bowl.pose.p - self.green_cube.pose.p,
                blue_cube_to_blue_bowl_pos=self.blue_bowl.pose.p - self.blue_cube.pose.p,
            )
        return obs

    def evaluate(self):
        pos_red_cube = self.red_cube.pose.p
        pos_green_cube = self.green_cube.pose.p
        pos_blue_cube = self.blue_cube.pose.p

        pos_red_bowl = self.red_bowl.pose.p
        pos_green_bowl = self.green_bowl.pose.p
        pos_blue_bowl = self.blue_bowl.pose.p

        def check_cube_in_bowl(cube_pos, bowl_pos):
            xy_dist = torch.linalg.norm(cube_pos[..., :2] - bowl_pos[..., :2], axis=1)
            z_diff = torch.abs(cube_pos[..., 2] - bowl_pos[..., 2])
            in_bowl = (xy_dist <= 0.08) & (z_diff <= 0.05)
            return in_bowl

        red_in_red_bowl = check_cube_in_bowl(pos_red_cube, pos_red_bowl)
        green_in_green_bowl = check_cube_in_bowl(pos_green_cube, pos_green_bowl)
        blue_in_blue_bowl = check_cube_in_bowl(pos_blue_cube, pos_blue_bowl)


        success = (
            red_in_red_bowl &
            green_in_green_bowl &
            blue_in_blue_bowl
        )

        return {
            "success": success,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):

        return 0

    def compute_normalized_dense_reward(
            self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    def get_instruction(self):
        return self.instruction