#!/usr/bin/env python3
"""
Robot Action — 发送控制指令到机械臂

用法:
    python action.py <x> <y> <z> <roll> <pitch> <yaw> <gripper>

参数:
    x, y, z      : TCP 位置（米，世界坐标系）
    roll, pitch, yaw : 姿态（弧度）
    gripper      : 夹爪状态 1.0=打开, 0.0=闭合

示例:
    python action.py 0.615 0.0 0.5 -3.14 0 0 1.0
    python action.py '[0.615, 0.0, 0.5, -3.14, 0, 0, 1.0]'
"""

import requests
import sys
import json

SERVER_URL = "http://127.0.0.1:8000"


def send_action(action):
    """发送 7 维动作指令"""
    if len(action) != 7:
        print(f"错误: 动作维度应为 7，实际为 {len(action)}")
        sys.exit(1)

    resp = requests.post(
        f"{SERVER_URL}/action",
        json={"action": [float(x) for x in action]},
        timeout=10,
    )

    if resp.status_code == 200:
        print(f"OK: {resp.text}")
    else:
        print(f"错误 (HTTP {resp.status_code}): {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 2 and (sys.argv[1].startswith("[") or sys.argv[1].startswith("{")):
        # JSON 数组格式
        data = json.loads(sys.argv[1])
        action = data if isinstance(data, list) else data.get("action", [])
    elif len(sys.argv) == 8:
        # 7 个独立参数
        action = [float(a) for a in sys.argv[1:]]
    else:
        print(__doc__)
        sys.exit(1)

    send_action(action)
