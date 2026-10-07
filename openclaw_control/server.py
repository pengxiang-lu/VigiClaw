from pathlib import Path
import sys
import argparse
import os

# Make direct execution from ``openclaw_control/`` behave like execution from
# the project root, so the sibling ``openclaw_task`` package is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ManiSkill appends ``data`` to this directory when loading external assets.
# Resolve it from this file so the server works from any current directory.
os.environ.setdefault("MS_ASSET_DIR", str(PROJECT_ROOT / "assets"))

import mani_skill.envs
import openclaw_task  # Registers the OpenClaw environments with ManiSkill/Gymnasium.
import queue
import threading
import time
import uuid
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import csv
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
MAX_FR = 50


class SimulationServer:
    """ManiSkill simulation server with an in-process HTTP control endpoint."""
    def __init__(self, host="127.0.0.1", port=8000, episode_nums=1, task="OpenclawMoveApple"):
        self.host = host
        self.port = port
        self.episode_nums = episode_nums  # Maximum number of episodes
        self.task = task
        self.completed_tasks = 0  # Number of completed tasks
        self.successful_tasks = 0  # Number of successful tasks
        self.task_in_progress = False  # Whether a task is in progress
        self.simulation_running = False
        self.env = None
        self.obs = None
        self.current_action = None
        self.last_action_success = None
        self.evaluation_result = None
        self.target_ee_pose = None
        self.position_threshold = 0.015  # 2cm threshold for position error
        self.action_arrived = False  # Flag to indicate if action has reached target
        self.arrival_message = None  # Message to send when action arrives
        self.last_gripper_state = None  # Track last gripper state
        self.gripper_change_time = None  # Track when gripper state changed
        self.gripper_arrival_delay = 0.5  # 1 second delay for gripper actions
        self.commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self.pending: dict[str, dict[str, Any]] = {}
        self.command_lock = threading.Lock()
        
        # CSV文件设置
        self.csv_file = "result.csv"
        self._init_csv_file()

    def start_simulation(self):
        """Start simulation thread"""
        self.simulation_running = True
        sim_thread = threading.Thread(target=self._run_simulation, daemon=True)
        sim_thread.start()
        print(f"Simulation thread started")

    def _run_simulation(self):
        """Run simulation environment main loop"""
        try:
            # Create Panda robot arm environment - Using end-effector pose control
            self.env = gym.make(self.task,
                                num_envs=1,
                                robot_uids="panda",
                                obs_mode="rgbd",
                                control_mode="pd_ee_pose",
                                render_mode="human",
                                sim_config={"control_freq": 50}
                                )
            self.obs, info = self.env.reset()
            self.initial_action = self._get_current_action()
            self.current_action = self._get_current_action()
            print(f"Initial action: {self.current_action}")

            print(f"Panda robot arm environment initialized: {self.task}")
            # Simulation main loop
            while self.simulation_running:
                if self.completed_tasks >= self.episode_nums:
                    print("[SERVER] Stopping simulation due to task completion...")
                    os._exit(0)
                start = time.time()

                action = self.current_action
                if hasattr(action, "cpu"):
                    action = action.cpu().numpy()
                elif hasattr(action, "numpy"):
                    action = action.numpy()
                elif not isinstance(action, np.ndarray):
                    action = np.array(action, dtype=np.float32)

                self.obs, reward, terminated, truncated, info = self.env.step(action)
                self._check_action_arrival()
                if self.env.render_mode == "human":
                    self.env.render()
                elapsed = time.time() - start
                diff = 1 / MAX_FR - elapsed
                if diff > 0:
                    time.sleep(diff)

                self._poll_local_command()

        except Exception as e:
            print(f"Simulation error: {e}")
            self.simulation_running = False
        finally:
            if self.env:
                self.env.close()

    def request_command(self, command: dict[str, Any], timeout: float = 30.0) -> Any:
        """Queue an HTTP command and wait for the simulation thread to handle it."""
        if not self.simulation_running or self.env is None:
            raise HTTPException(status_code=503, detail="Simulation server is not ready")
        request_id = str(uuid.uuid4())
        event = threading.Event()
        with self.command_lock:
            self.pending[request_id] = {"event": event, "result": None}
        self.commands.put({**command, "request_id": request_id})
        if not event.wait(timeout):
            with self.command_lock:
                self.pending.pop(request_id, None)
            raise HTTPException(status_code=504, detail="Simulation response timed out")
        with self.command_lock:
            pending = self.pending.pop(request_id, None)
        return pending["result"] if pending is not None else None

    def _poll_local_command(self):
        """Process one command queued by the local HTTP API."""
        try:
            command = self.commands.get_nowait()
            result = self._handle_remote_command(command)
        except queue.Empty:
            return
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
        with self.command_lock:
            pending = self.pending.get(command["request_id"])
            if pending is not None:
                pending["result"] = result
                pending["event"].set()

    @staticmethod
    def _tensor_to_list(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy().tolist()
        if hasattr(value, "numpy"):
            return value.numpy().tolist()
        return np.asarray(value).tolist()

    def _handle_remote_command(self, command):
        """Execute a command received from the human-control server."""
        command_type = command.get("type")
        if command_type in {"action", "reset", "end_task"}:
            self._process_command(command)
            return {"success": self.last_action_success is not False}
        if command_type == "state":
            sensor_data = self.obs["sensor_data"]
            base_camera = sensor_data["base_camera"]
            hand_camera = sensor_data.get("hand_camera")
            current_action = self._get_current_action().tolist()
            current_action[0] -= 0.615
            return {
                "base_camera_rgb": self._tensor_to_list(base_camera["rgb"]),
                "base_camera_depth": self._tensor_to_list(base_camera["depth"]),
                "hand_camera_rgb": self._tensor_to_list(hand_camera["rgb"]) if hand_camera else None,
                "hand_camera_depth": self._tensor_to_list(hand_camera["depth"]) if hand_camera else None,
                "current_action": current_action,
                "action_arrived": self.action_arrived,
                "arrival_message": self.arrival_message,
                "timestamp": time.time(),
            }
        if command_type == "arrival_status":
            current_action = self._get_current_action().tolist()
            current_action[0] -= 0.615
            return {
                "action_arrived": self.action_arrived,
                "arrival_message": self.arrival_message,
                "current_action": current_action,
                "timestamp": time.time(),
            }
        if command_type == "camera_params":
            params = self.obs.get("sensor_param", {})
            base = params.get("base_camera")
            hand = params.get("hand_camera")
            if base is None:
                raise RuntimeError("Base camera parameters are unavailable")
            return {
                "base_extrinsic_cv": self._tensor_to_list(base["extrinsic_cv"]),
                "base_intrinsic_cv": self._tensor_to_list(base["intrinsic_cv"]),
                "hand_extrinsic_cv": self._tensor_to_list(hand["extrinsic_cv"]) if hand else None,
                "hand_intrinsic_cv": self._tensor_to_list(hand["intrinsic_cv"]) if hand else None,
                "timestamp": time.time(),
            }
        if command_type == "evaluation":
            evaluation = self.env.unwrapped.evaluate()
            return {key: self._tensor_to_list(value) if torch.is_tensor(value) else value for key, value in evaluation.items()}
        if command_type == "task_stats":
            success_rate = (
                self.successful_tasks / self.completed_tasks * 100
                if self.completed_tasks
                else 0.0
            )
            return {
                "episode_nums": self.episode_nums,
                "max_tasks": self.episode_nums,
                "completed_tasks": self.completed_tasks,
                "successful_tasks": self.successful_tasks,
                "success_rate": round(success_rate, 2),
                "task_in_progress": self.task_in_progress,
                "remaining_tasks": max(0, self.episode_nums - self.completed_tasks),
            }
        if command_type == "instruction":
            return {"task_instruction": self.env.unwrapped.get_instruction()}
        if command_type == "object_positions":
            return self._get_object_positions()
        raise ValueError(f"Unknown remote command: {command_type}")

    def _get_object_positions(self):
        """Return poses for the common OpenClaw scene objects."""
        def pose_to_list(obj):
            if obj is None:
                return None
            position = np.asarray(self._tensor_to_list(obj.pose.p)).reshape(-1)[:3]
            quaternion = np.asarray(self._tensor_to_list(obj.pose.q)).reshape(-1)[:4]
            w, x, y, z = quaternion / np.linalg.norm(quaternion)
            roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            pitch_value = 2 * (w * y - z * x)
            pitch = np.arcsin(np.clip(pitch_value, -1, 1))
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            return [*(round(float(v), 3) for v in position), round(float(pitch), 3), round(float(roll), 3), round(float(yaw), 3)]

        env = self.env.unwrapped
        return {
            "red_cube": pose_to_list(getattr(env, "red_cube", None)),
            "green_cube": pose_to_list(getattr(env, "green_cube", None)),
            "blue_cube": pose_to_list(getattr(env, "blue_cube", None)),
            "yellow_cube": pose_to_list(getattr(env, "yellow_cube", None)),
            "timestamp": round(time.time(), 3),
        }

    def _get_current_action(self):
        """Get current 7-dim action (with actual gripper state)"""
        try:
            # Return the last executed action if available (preserves gripper state)
            if hasattr(self, 'current_action') and self.current_action is not None:
                return self.current_action
            
            # Otherwise, read from environment (e.g., after reset)
            current_ee_pose = self.obs["extra"]["tcp_pose"][0]

            # Convert to numpy array
            if hasattr(current_ee_pose, 'cpu'):
                current_ee_pose = current_ee_pose.cpu().numpy()
            elif hasattr(current_ee_pose, 'numpy'):
                current_ee_pose = current_ee_pose.numpy()
            else:
                current_ee_pose = np.array(current_ee_pose)

            # Handle 2D tensor shape (1, 7) -> convert to 1D array (7,)
            if len(current_ee_pose.shape) == 2 and current_ee_pose.shape[0] == 1:
                current_ee_pose = current_ee_pose[0]

            # Extract position
            position = current_ee_pose[0:3]  # [x, y, z]

            # Get gripper state (default to open: 1.0)
            gripper_state = 1.0

            # Combine to 7-dim array: [x, y, z, roll, pitch, yaw, gripper]
            action = np.array([
                position[0] + 0.615, position[1], position[2],
                -3.14, 0, 0,
                gripper_state
            ], dtype=np.float32)
            # 将world_frame转换为root_frame
            return action
        except Exception as e:
            print(f"Error getting current action: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _check_reachability(self, target_pos, target_rot):
        """Check if the robot arm can reach the target position using IK solver"""
        try:
            from mani_skill.utils.geometry.rotation_conversions import (
                euler_angles_to_matrix,
                matrix_to_quaternion
            )
            from mani_skill.utils.structs.pose import Pose
            import torch
            
            # Get the arm controller
            agent = self.env.unwrapped.agent
            arm_controller = agent.controller.controllers["arm"]
            
            # Get current joint positions
            current_qpos = agent.robot.get_qpos()
            
            # Convert target rotation (roll, pitch, yaw) to quaternion
            target_euler = torch.tensor(target_rot, dtype=torch.float32, device=current_qpos.device)
            target_rot_matrix = euler_angles_to_matrix(target_euler.unsqueeze(0), "XYZ")
            target_quat = matrix_to_quaternion(target_rot_matrix)
            
            # Convert target position to tensor
            target_pos_tensor = torch.tensor(target_pos, dtype=torch.float32, device=current_qpos.device).unsqueeze(0)
            
            # Create Pose object for IK
            target_pose = Pose.create_from_pq(target_pos_tensor, target_quat)
            
            # Use IK to compute joint positions
            ik_result = arm_controller.kinematics.compute_ik(
                pose=target_pose,
                q0=current_qpos,
                is_delta_pose=False,
                solver_config=dict(type="levenberg_marquardt", alpha=1.0)
            )
            
            # Check if IK was successful
            if ik_result is not None:
                print(f"IK successful: Target position {target_pos} is reachable")
                return True, ik_result
            else:
                print(f"IK failed: Target position {target_pos} is not reachable")
                return False, None
                
        except Exception as e:
            print(f"Error checking reachability: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def _init_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['task_id', 'timestamp', 'instruction', 'success'])
                print(f"[CSV] Created new result file: {self.csv_file}")
            except Exception as e:
                print(f"[CSV ERROR] Failed to create CSV file: {e}")
    
    def _save_task_result(self, task_id, instruction, success):
        """Save task result to CSV file"""
        try:
            # Prepare data row
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = [
                task_id,
                timestamp,
                instruction,
                success
            ]
            
            # Append to CSV file
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            print(f"[CSV] Task result saved to {self.csv_file}")
            
        except Exception as e:
            print(f"[CSV ERROR] Failed to save task result: {e}")
    
    def _get_task_instruction(self):
        """Get current task instruction"""
        try:
            instruction = self.env.unwrapped.get_instruction()
            return instruction
        except Exception as e:
            print(f"[INSTRUCTION ERROR] Failed to get task instruction: {e}")
            return "Unknown instruction"
    
    def _end_current_task(self):
        """End current task and evaluate success"""
        try:
            # Get task instruction
            instruction = self._get_task_instruction()
            
            # Evaluate task success
            self.evaluation_result = self.env.unwrapped.evaluate()
            
            # Check if task was successful
            task_success = False
            if self.evaluation_result is not None:
                # Check for success indicators in evaluation result
                if isinstance(self.evaluation_result, dict):
                    # Look for success-related keys
                    success_keys = ['success', 'is_success', 'task_success']
                    for key in success_keys:
                        if key in self.evaluation_result:
                            task_success = bool(self.evaluation_result[key])
                            break
                    
                    # If no explicit success key, check for positive reward
                    if not any(key in self.evaluation_result for key in success_keys):
                        if 'reward' in self.evaluation_result:
                            task_success = self.evaluation_result['reward'] > 0
            
            # Update statistics
            self.completed_tasks += 1
            if task_success:
                self.successful_tasks += 1
                print(f"[TASK COMPLETED] Task {self.completed_tasks}/{self.episode_nums} SUCCESSFUL")
            else:
                print(f"[TASK COMPLETED] Task {self.completed_tasks}/{self.episode_nums} FAILED")
            
            # Save task result to CSV
            self._save_task_result(self.completed_tasks, instruction, task_success)
            
            # Calculate success rate
            if self.completed_tasks > 0:
                success_rate = (self.successful_tasks / self.completed_tasks) * 100
                print(f"[STATISTICS] Success rate: {success_rate:.2f}% ({self.successful_tasks}/{self.completed_tasks})")
            
            # Check if max tasks reached
            if self.completed_tasks >= self.episode_nums:
                print(f"[PROGRAM COMPLETE] All {self.episode_nums} episodes completed!")
                print(f"[FINAL STATISTICS] Success rate: {success_rate:.2f}% ({self.successful_tasks}/{self.completed_tasks})")
                # Stop the entire script
                print("[SERVER] Stopping simulation due to task completion...")
                os._exit(0)
            
            # Reset environment for next task
            self.obs, info = self.env.reset()
            self.current_action = self.initial_action
            self.target_ee_pose = None
            self.action_arrived = False
            self.arrival_message = None
            self.last_gripper_state = None
            self.gripper_change_time = None
            
        except Exception as e:
            print(f"Error ending task: {e}")
            import traceback
            traceback.print_exc()
    
    def _check_action_arrival(self):
        """Check if the robot arm has reached the target position"""
        if self.target_ee_pose is None or self.action_arrived:
            return
        
        try:
            # Get current end-effector position
            current_ee_pose = self.obs["extra"]["tcp_pose"][0]
            
            # Convert to numpy array
            if hasattr(current_ee_pose, 'cpu'):
                current_ee_pose = current_ee_pose.cpu().numpy()
            elif hasattr(current_ee_pose, 'numpy'):
                current_ee_pose = current_ee_pose.numpy()
            else:
                current_ee_pose = np.array(current_ee_pose)
            
            # Handle 2D tensor shape (1, 7) -> convert to 1D array (7,)
            if len(current_ee_pose.shape) == 2 and current_ee_pose.shape[0] == 1:
                current_ee_pose = current_ee_pose[0]
            
            # Extract current position (world frame)
            current_pos_world = np.array([
                current_ee_pose[0],
                current_ee_pose[1],
                current_ee_pose[2]
            ])
            
            # Convert target position to world frame (subtract the offset)
            target_pos_world = np.array([
                self.target_ee_pose[0] - 0.615,
                self.target_ee_pose[1],
                self.target_ee_pose[2]
            ])
            
            # Calculate distance to target
            distance = np.linalg.norm(current_pos_world - target_pos_world)
            
            # Check if gripper state changed - if so, wait for delay before checking arrival
            if self.gripper_change_time is not None:
                time_since_gripper_change = time.time() - self.gripper_change_time
                if time_since_gripper_change < self.gripper_arrival_delay:
                    # Still in delay period, skip arrival check
                    return
            
            # Check if within threshold
            if distance <= self.position_threshold:
                self.action_arrived = True
                self.arrival_message = f"Action reached target position! Distance: {distance:.4f}m (threshold: {self.position_threshold}m)"
                print(f"[ARRIVAL] {self.arrival_message}")
                
        except Exception as e:
            print(f"Error checking action arrival: {e}")
            import traceback
            traceback.print_exc()

    def _process_command(self, command):
        """Process control commands"""
        try:
            cmd_type = command.get('type', '')

            if cmd_type == 'action':
                # Execute 7-dimensional action directly
                action_data = command.get('action', [0, 0, 0, 0, 0, 0, 0])

                if len(action_data) != 7:
                    print(f"Invalid action dimension: expected 7, got {len(action_data)}")
                    return

                # Check if gripper state changed
                current_gripper_state = action_data[6]
                if self.last_gripper_state is not None and abs(current_gripper_state - self.last_gripper_state) > 0.1:
                    # Gripper state changed significantly
                    self.gripper_change_time = time.time()
                    print(f"[GRIPPER] Gripper state changed from {self.last_gripper_state} to {current_gripper_state}, waiting {self.gripper_arrival_delay}s before checking arrival")
                else:
                    # Gripper state did not change significantly
                    self.gripper_change_time = None
                
                self.last_gripper_state = current_gripper_state

                # Set the current action to execute
                self.current_action = np.array(action_data, dtype=np.float32)
                self.current_action[0] += 0.615                                 # 将world_frame转换为root_frame
                
                # Extract target position and rotation (with offset for IK)
                target_pos = [self.current_action[0], self.current_action[1], self.current_action[2]]
                target_rot = [action_data[3], action_data[4], action_data[5]]
                
                # Check if target is reachable before executing action
                is_reachable, ik_result = self._check_reachability(target_pos, target_rot)
                self.last_action_success = is_reachable
                
                if not is_reachable:
                    print(f"Cannot reach target position: {target_pos}")
                    return

                # Save target end-effector position for motion success detection
                self.target_ee_pose = np.array([
                    self.current_action[0],
                    self.current_action[1], 
                    self.current_action[2]
                ])
                # Reset arrival flag when new action is set
                self.action_arrived = False
                self.arrival_message = None
                print(f"Set new action: {self.current_action}")
            elif cmd_type == 'reset':
                # Reset the simulation environment
                self.obs, info = self.env.reset()
                self.current_action = self.initial_action
                self.target_ee_pose = None
                self.last_action_success = None
                self.action_arrived = False
                self.arrival_message = None
                self.last_gripper_state = None
                self.gripper_change_time = None
                print("Environment reset")
            elif cmd_type == 'end_task':
                # End current task and evaluate success
                self._end_current_task()

        except Exception as e:
            print(f"Command processing error: {e}")

    def run(self):
        """Start the simulation and its local HTTP control API."""
        print("Starting ManiSkill simulation server...")

        self.start_simulation()

        uvicorn.run(create_app(self), host=self.host, port=self.port)


class ActionCommand(BaseModel):
    action: list[float]


def create_app(server: SimulationServer) -> FastAPI:
    """Create the HTTP API served by the same process as the simulation."""
    app = FastAPI(title="VigiClaw Simulation Server", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {"message": "VigiClaw simulation server running", "task": server.task}

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "simulation_running": server.simulation_running,
            "environment_ready": server.env is not None,
            "task": server.task,
        }

    @app.post("/action")
    def action(command: ActionCommand):
        if len(command.action) != 7:
            raise HTTPException(status_code=400, detail="action must contain 7 values")
        result = server.request_command({"type": "action", "action": command.action}) or {}
        return {"status": "success", "message": "Action applied", **result}

    @app.get("/arrival_status")
    def arrival_status():
        return server.request_command({"type": "arrival_status"})

    @app.get("/state")
    def state():
        return server.request_command({"type": "state"})

    @app.get("/camera_params")
    def camera_params():
        return server.request_command({"type": "camera_params"})

    @app.post("/reset")
    def reset():
        return {"status": "success", **(server.request_command({"type": "reset"}) or {})}

    @app.post("/end_task")
    def end_task():
        return {"status": "success", **(server.request_command({"type": "end_task"}) or {})}

    @app.get("/task_stats")
    def task_stats():
        return {"status": "success", "stats": server.request_command({"type": "task_stats"})}

    @app.get("/evaluation")
    def evaluation():
        return {
            "status": "success",
            "evaluation": server.request_command({"type": "evaluation"}),
            "timestamp": time.time(),
        }

    @app.get("/instruction")
    def instruction():
        return server.request_command({"type": "instruction"})

    @app.get("/object_positions")
    def object_positions():
        return {
            "status": "success",
            "poses": server.request_command({"type": "object_positions"}),
            "message": "Object poses retrieved successfully",
        }

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start the VigiClaw ManiSkill simulation server."
    )
    parser.add_argument(
        "--task",
        default="OpenclawMoveApple",
        choices=openclaw_task.TASK_ENV_IDS,
        help="OpenClaw ManiSkill task to run (default: OpenclawMoveApple).",
    )
    parser.add_argument(
        "--episode_nums",
        type=int,
        default=1,
        metavar="N",
        help="Number of episodes to run before stopping (default: 1).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP bind port.")
    args = parser.parse_args()

    if args.episode_nums < 1:
        parser.error("--episode_nums must be at least 1")

    server = SimulationServer(
        host=args.host,
        port=args.port,
        task=args.task,
        episode_nums=args.episode_nums,
    )
    server.run()
