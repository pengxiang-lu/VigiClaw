from pathlib import Path
import os

# Use the project-level asset directory regardless of the current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MS_ASSET_DIR", str(PROJECT_ROOT / "assets"))

import gymnasium as gym
import torch
import threading
import time
import numpy as np
from pynput import keyboard

import mani_skill.envs

MAX_FR = 50

class RobotControl:
    """Robot Control with keyboard input"""

    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        self.simulation_running = False
        self.env = None
        self.obs = None
        self.current_action = None
        self.step_size = 0.01
        self.rotation_step = 0.05
        self.gripper_state = 1.0

    def start_simulation(self):
        """Start simulation environment"""
        self.simulation_running = True
        sim_thread = threading.Thread(target=self._run_simulation, daemon=True)
        sim_thread.start()
        print(f"Simulation thread started")

    def _run_simulation(self):
        """Run simulation environment main loop"""
        try:
            self.env = gym.make(
                "PickCube-v1",
                num_envs=1,
                obs_mode="rgbd",
                control_mode="pd_ee_pose",
                render_mode=self.render_mode,
                robot_uids="panda_wristcam"
            )
            self.obs, info = self.env.reset(seed=0)
            self.current_action = self._get_current_action()
            print(f"Initial action: {self.current_action}")
            print("Panda robot arm environment initialized")

            while self.simulation_running:
                start = time.time()
                action = self.current_action
                if hasattr(action, 'cpu'):
                    action = action.cpu().numpy()
                elif hasattr(action, 'numpy'):
                    action = action.numpy()
                elif not isinstance(action, np.ndarray):
                    action = np.array(action, dtype=np.float32)

                self.obs, reward, terminated, truncated, info = self.env.step(action)

                if self.env.render_mode == "human":
                    self.env.render()
                    elapsed = time.time() - start
                    diff = 1 / MAX_FR - elapsed
                    if diff > 0:
                        time.sleep(diff)

        except Exception as e:
            print(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()
            self.simulation_running = False
        finally:
            if self.env:
                self.env.close()

    def _get_current_action(self):
        """Get current 7-dim action"""
        try:
            if hasattr(self, 'current_action') and self.current_action is not None:
                return self.current_action

            current_ee_pose = self.obs["extra"]["tcp_pose"][0]

            if hasattr(current_ee_pose, 'cpu'):
                current_ee_pose = current_ee_pose.cpu().numpy()
            elif hasattr(current_ee_pose, 'numpy'):
                current_ee_pose = current_ee_pose.numpy()
            else:
                current_ee_pose = np.array(current_ee_pose)

            if len(current_ee_pose.shape) == 2 and current_ee_pose.shape[0] == 1:
                current_ee_pose = current_ee_pose[0]

            position = current_ee_pose[0:3]
            gripper_state = 1.0

            action = np.array([
                position[0] + 0.615, position[1], position[2],
                -3.14, 0, 0,
                gripper_state
            ], dtype=np.float32)
            return action
        except Exception as e:
            print(f"Error getting current action: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def on_press(self, key):
        """Handle key press events"""
        try:
            if key == keyboard.Key.esc:
                print("\nStopping...")
                self.simulation_running = False
                return False

            action_changed = False
            new_action = self.current_action.copy()

            if key == keyboard.KeyCode.from_char('2'):
                new_action[0] += self.step_size
                action_changed = True
            elif key == keyboard.KeyCode.from_char('8'):
                new_action[0] -= self.step_size
                action_changed = True
            elif key == keyboard.KeyCode.from_char('4'):
                new_action[1] -= self.step_size
                action_changed = True
            elif key == keyboard.KeyCode.from_char('6'):
                new_action[1] += self.step_size
                action_changed = True
            elif key == keyboard.KeyCode.from_char('7'):
                new_action[2] += self.step_size
                action_changed = True
            elif key == keyboard.KeyCode.from_char('1'):
                new_action[2] -= self.step_size
                action_changed = True
            elif key == keyboard.KeyCode.from_char('o'):
                new_action[3] += self.rotation_step
                action_changed = True
            elif key == keyboard.KeyCode.from_char('l'):
                new_action[3] -= self.rotation_step
                action_changed = True
            elif key == keyboard.KeyCode.from_char('u'):
                new_action[4] += self.rotation_step
                action_changed = True
            elif key == keyboard.KeyCode.from_char('j'):
                new_action[4] -= self.rotation_step
                action_changed = True
            elif key == keyboard.KeyCode.from_char('i'):
                new_action[5] += self.rotation_step
                action_changed = True
            elif key == keyboard.KeyCode.from_char('k'):
                new_action[5] -= self.rotation_step
                action_changed = True
            elif key == keyboard.Key.space:
                self.gripper_state = -1.0 if self.gripper_state > 0 else 1.0
                new_action[6] = self.gripper_state
                action_changed = True
                print(f"\nGripper: {'Open' if self.gripper_state > 0.5 else 'Closed'}")
            
            if action_changed:
                self.current_action = new_action.copy()
                self.display_current_action()

        except AttributeError:
            pass

    def display_current_action(self):
        """Display current action state"""
        print(f"\rAction: x={self.current_action[0]:.3f}, y={self.current_action[1]:.3f}, "
              f"z={self.current_action[2]:.3f}, roll={self.current_action[3]:.2f}, "
              f"pitch={self.current_action[4]:.2f}, yaw={self.current_action[5]:.2f}, "
              f"gripper={'Open' if self.current_action[6] > 0.5 else 'Closed'}", end="")

    def display_controls(self):
        """Display control instructions"""
        print("=" * 60)
        print("Robot Arm Keyboard Control")
        print("=" * 60)
        print("Position Control:")
        print("  2 - Move X axis forward")
        print("  8 - Move X axis backward")
        print("  4 - Move Y axis left")
        print("  6 - Move Y axis right")
        print("  7 - Move Z axis up")
        print("  1 - Move Z axis down")
        print("\nRotation Control:")
        print("  O - Roll rotation (positive)")
        print("  L - Roll rotation (negative)")
        print("  U - Pitch rotation (positive)")
        print("  J - Pitch rotation (negative)")
        print("  I - Yaw rotation (positive)")
        print("  K - Yaw rotation (negative)")
        print("\nGripper Control:")
        print("  Space - Toggle gripper (open/close)")
        print("\nOther:")
        print("  ESC - Exit")
        print("=" * 60)

    def start_keyboard_control(self):
        """Start keyboard listener"""
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        print("Keyboard control started")

    def run(self):
        """Run the control system"""
        print("=" * 60)
        print("Robot Arm Control")
        print("=" * 60)

        self.start_simulation()
        time.sleep(1)

        self.start_keyboard_control()

        while self.simulation_running:
            time.sleep(0.02)

        print("\nShutting down...")
        self.simulation_running = False


if __name__ == "__main__":
    render_mode = "human"
    control = RobotControl(render_mode=render_mode)
    control.run()
