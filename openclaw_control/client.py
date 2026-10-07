#!/usr/bin/env python3
"""
Robot Control Client
Connects to server and sends 7-dim action control commands
"""
import numpy as np
import requests
import json
import time
import threading
import sys


class RobotControlClient:
    """Robot Control Client"""

    def __init__(self, server_url="http://127.0.0.1:8000"):
        self.server_url = server_url
        self.session = requests.Session()
        self.task_count = 0  # Track number of tasks completed by this client

    def check_connection(self):
        """Check server connection"""
        try:
            response = self.session.get(f"{self.server_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"Server connection OK: {data}")
                return True
            else:
                print(f"Server response error: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to server: {e}")
            return False

    def check_health(self):
        """Check server health"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"Server health: {data}")
                return True
            else:
                print(f"Health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return False

    def reset_environment(self):
        """Reset the simulation environment"""
        try:
            response = self.session.post(f"{self.server_url}/reset", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"Environment reset: {data}")
                return True
            else:
                print(f"Reset failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Reset failed: {e}")
            return False
    
    def end_task(self):
        """End current task and evaluate success"""
        try:
            response = self.session.post(f"{self.server_url}/end_task", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"Task ended: {data}")
                self.task_count += 1
                return True
            else:
                print(f"End task failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"End task failed: {e}")
            return False
    
    def get_task_stats(self):
        """Get task statistics from server"""
        try:
            response = self.session.get(f"{self.server_url}/task_stats", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    stats = data.get("stats", {})
                    return stats
                else:
                    print(f"Failed to get stats: {data}")
                    return None
            else:
                print(f"Stats request failed: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Stats request failed: {e}")
            return None
    
    def get_object_positions(self):
        """Get poses (position and orientation) of all objects in the simulation environment"""
        try:
            response = self.session.get(f"{self.server_url}/object_positions", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    poses = data.get("poses", {})
                    return poses
                else:
                    print(f"Failed to get object poses: {data}")
                    return None
            else:
                print(f"Object poses request failed: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Object poses request failed: {e}")
            return None
    
    def print_object_positions(self):
        """Print poses (position and orientation) of all objects in the simulation environment"""
        poses = self.get_object_positions()
        if poses:
            print("\n=== Object Poses [x, y, z, pitch, roll, yaw] ===")
            
            def format_pose(pose_data):
                if pose_data is None:
                    return "N/A"
                if len(pose_data) >= 6:
                    return f"[{pose_data[0]:.4f}, {pose_data[1]:.4f}, {pose_data[2]:.4f}, {pose_data[3]:.4f}, {pose_data[4]:.4f}, {pose_data[5]:.4f}]"
                return str(pose_data)
            
            print(f"Red Cube:    {format_pose(poses.get('red_cube'))}")
            print(f"Green Cube:  {format_pose(poses.get('green_cube'))}")
            print(f"Blue Cube:   {format_pose(poses.get('blue_cube'))}")
            print(f"Yellow Cube: {format_pose(poses.get('yellow_cube'))}")
            print(f"Timestamp:   {poses.get('timestamp', 'N/A')}")
            print("=====================================================")
        else:
            print("Failed to get object poses")
    
    def print_task_stats(self):
        """Print formatted task statistics"""
        stats = self.get_task_stats()
        if stats:
            print("\n" + "="*50)
            print("TASK STATISTICS")
            print("="*50)
            print(f"Maximum Tasks: {stats.get('max_tasks', 0)}")
            print(f"Completed Tasks: {stats.get('completed_tasks', 0)}")
            print(f"Successful Tasks: {stats.get('successful_tasks', 0)}")
            print(f"Success Rate: {stats.get('success_rate', 0)}%")
            print(f"Remaining Tasks: {stats.get('remaining_tasks', 0)}")
            print(f"Task In Progress: {'Yes' if stats.get('task_in_progress', False) else 'No'}")
            print("="*50 + "\n")
        else:
            print("Failed to retrieve task statistics")

    def send_action(self, action, wait_for_arrival=True, timeout=5.0):
        """Send 7-dim action [x, y, z, roll, pitch, yaw, gripper] to control the robot
        
        Args:
            action: 7-dim action array
            wait_for_arrival: If True, wait for the action to reach target position
            timeout: Maximum time to wait for arrival (in seconds)
        
        Returns:
            True if action succeeded and reached target (if wait_for_arrival=True), False otherwise
        """
        if len(action) != 7:
            print("Invalid action: must be 7-dimensional [x, y, z, roll, pitch, yaw, gripper]")
            return False

        try:
            command = {"action": action}
            response = self.session.post(
                f"{self.server_url}/action",
                json=command,
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("success", True):
                    print(f"Action sent successfully: {action}")
                    
                    # Wait for arrival if requested
                    if wait_for_arrival:
                        return self._wait_for_arrival(timeout)
                    else:
                        return True
                else:
                    print(f"Action failed: {result.get('message', 'Unknown error')}")
                    return False
            else:
                print(f"Action request failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Action request failed: {e}")
            return False

    def _wait_for_arrival(self, timeout=5.0):
        """Wait for the action to reach target position
        
        Args:
            timeout: Maximum time to wait (in seconds)
        
        Returns:
            True if arrived within timeout, False otherwise
        """
        start_time = time.time()
        check_interval = 0.1  # Check every 100ms
        
        while True:
            elapsed_time = time.time() - start_time
            
            # Check timeout
            if elapsed_time >= timeout:
                print(f"Timeout: Action did not reach target within {timeout}s")
                return False
            
            # Get arrival status (lightweight, no images)
            status = self._get_arrival_status()
            if status is None:
                print("Failed to get arrival status while waiting")
                return False
            
            # Check if arrived
            if status.get('action_arrived', False):
                print(f"Action reached target position in {elapsed_time:.2f}s")
                return True
            
            # Wait before next check
            time.sleep(check_interval)

    def _get_arrival_status(self):
        """Get only the arrival status without image data (lightweight)"""
        try:
            response = self.session.get(f"{self.server_url}/arrival_status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get arrival status: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Getting arrival status failed: {e}")
            return None

    def get_state(self, verbose=True):
        """Get robot state including cameras and end effector pose
        
        Args:
            verbose: If True, print detailed information. If False, return state silently.
        
        Returns:
            State dictionary or None if failed
        """
        try:
            response = self.session.get(f"{self.server_url}/state", timeout=10)
            if response.status_code == 200:
                state = response.json()
                
                if verbose:
                    print("Successfully retrieved robot state")
                    print(
                        f"Current action: {state['current_action'][:3]} (position), {state['current_action'][3:6]} (orientation), {state['current_action'][6]} (gripper)")

                    # Show arrival status
                    if 'action_arrived' in state:
                        if state['action_arrived']:
                            print("=" * 60)
                            print("✓ ACTION REACHED TARGET POSITION!")
                            print(f"  {state.get('arrival_message', 'No message')}")
                            print("=" * 60)
                        else:
                            print("  Action still moving to target position...")

                    # Show actual shapes of the data
                    base_rgb_data = state['base_camera_rgb']
                    base_depth_data = state['base_camera_depth']
                    hand_rgb_data = state['hand_camera_rgb']
                    hand_depth_data = state['hand_camera_depth']
                    # Determine actual shapes
                    def get_shape(data):
                        if isinstance(data, list):
                            shape = []
                            current = data
                            while isinstance(current, list) and len(current) > 0:
                                shape.append(len(current))
                                current = current[0] if len(current) > 0 else None
                            return 'x'.join(map(str, shape))
                        return "unknown"

                    print(f"Base camera RGB shape: {get_shape(base_rgb_data)}")
                    print(f"Base camera depth shape: {get_shape(base_depth_data)}")
                    print(f"Hand camera RGB shape: {get_shape(hand_rgb_data)}")
                    print(f"Hand camera depth shape: {get_shape(hand_depth_data)}")
                return state
            else:
                if verbose:
                    print(f"Failed to get state: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"Getting state failed: {e}")
            return None

    def get_camera_params(self):
        """Get camera intrinsic and extrinsic parameters"""
        try:
            response = self.session.get(f"{self.server_url}/camera_params", timeout=10)
            if response.status_code == 200:
                camera_params = response.json()
                print("Successfully retrieved camera parameters")
                print(f"Base Extrinsic matrix (world to camera): {np.array(camera_params['base_extrinsic_cv']).shape}")
                print(f"Base Intrinsic matrix: {camera_params['base_intrinsic_cv']}")
                print(f"Hand Extrinsic matrix (world to camera): {np.array(camera_params['hand_extrinsic_cv']).shape}")
                print(f"Hand Intrinsic matrix: {np.array(camera_params['hand_intrinsic_cv']).shape}")
                return camera_params
            else:
                print(f"Failed to get camera parameters: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Getting camera parameters failed: {e}")
            return None

    def get_evaluation(self):
        """Get task evaluation results"""
        try:
            response = self.session.get(f"{self.server_url}/evaluation", timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    evaluation = result.get("evaluation", {})
                    for key, value in evaluation.items():
                        if isinstance(value, list) and len(value) > 0:
                            print(f"{key}: {value[0]}")
                        else:
                            print(f"{key}: {value}")
                    return evaluation
                else:
                    print(f"No evaluation data: {result.get('message', 'Unknown error')}")
                    return None
            else:
                print(f"Failed to get evaluation: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Getting evaluation failed: {e}")
            return None

    def get_instruction(self):
        """Get task instruction"""
        try:
            response = self.session.get(f"{self.server_url}/instruction", timeout=10)
            if response.status_code == 200:
                result = response.json()
                instruction = result.get("task_instruction", "")
                print(f"Instruction: {instruction}")
                return {"task_instruction": instruction}
            else:
                print(f"Failed to get instruction: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Getting instruction failed: {e}")
            return None

    def interactive_control(self):
        """Interactive control interface"""
        print("=" * 50)
        print("Robot Arm Control Client")
        print("Send 7-dimensional action [x, y, z, roll, pitch, yaw, gripper]")
        print("=" * 50)

        if not self.check_connection():
            print("Cannot connect to server, please ensure server is running")
            return

        current_action = [-0.3, 0, 0.3, -3.14, 0, 0, 1]
        print(f"Initial action: {current_action}")

        while True:
            print("\nCommands:")
            print("1. Send 7-dim action")
            print("2. Get state (cameras and end effector)")
            print("3. Get camera parameters (intrinsic & extrinsic)")
            print("4. Get task evaluation")
            print("5. Check server health")
            print("6. Reset environment")
            print("7. Get task Instruction")
            print("8. End current task")
            print("9. View task statistics")
            print("10. Get object positions")
            print("11. Exit")

            choice = input("Enter command number: ").strip()

            if choice == "1":
                try:
                    print("Enter 7 values for [x, y, z, roll, pitch, yaw, gripper]:")
                    print("(Press Enter to keep previous value)")
                    
                    x_input = input(f"X coordinate [current={current_action[0]}]: ").strip()
                    x = float(x_input) if x_input else current_action[0]
                    
                    y_input = input(f"Y coordinate [current={current_action[1]}]: ").strip()
                    y = float(y_input) if y_input else current_action[1]
                    
                    z_input = input(f"Z coordinate [current={current_action[2]}]: ").strip()
                    z = float(z_input) if z_input else current_action[2]
                    
                    roll_input = input(f"Roll [current={current_action[3]}]: ").strip()
                    roll = float(roll_input) if roll_input else current_action[3]
                    
                    pitch_input = input(f"Pitch [current={current_action[4]}]: ").strip()
                    pitch = float(pitch_input) if pitch_input else current_action[4]
                    
                    yaw_input = input(f"Yaw [current={current_action[5]}]: ").strip()
                    yaw = float(yaw_input) if yaw_input else current_action[5]
                    
                    gripper_input = input(f"Gripper (0=closed, 1=open) [current={current_action[6]}]: ").strip()
                    gripper = float(gripper_input) if gripper_input else current_action[6]

                    action = [x, y, z, roll, pitch, yaw, gripper]
                    current_action = action
                    self.send_action(action)
                except ValueError:
                    print("Invalid input, please enter numbers")

            elif choice == "2":
                self.get_state()

            elif choice == "3":
                self.get_camera_params()

            elif choice == "4":
                self.get_evaluation()

            elif choice == "5":
                self.check_health()

            elif choice == "6":
                self.reset_environment()
            elif choice == "7":
                self.get_instruction()
            elif choice == "8":
                self.end_task()
            elif choice == "9":
                self.print_task_stats()
            elif choice == "10":
                self.print_object_positions()
            elif choice == "11":
                print("Exiting control client")
                break
            else:
                print("Invalid choice, please try again")


if __name__ == "__main__":
    client = RobotControlClient()
    client.interactive_control()