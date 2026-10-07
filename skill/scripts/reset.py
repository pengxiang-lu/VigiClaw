#!/usr/bin/env python3
"""
Robot Reset — 重置仿真环境到初始状态

用法:
    python reset.py

示例:
    python reset.py
"""

import requests
import sys

SERVER_URL = "http://127.0.0.1:8000"


def reset_env():
    """重置环境"""
    resp = requests.post(f"{SERVER_URL}/reset", timeout=5)

    if resp.status_code == 200:
        print("OK: 环境已重置")
    else:
        print(f"错误 (HTTP {resp.status_code}): {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    reset_env()
