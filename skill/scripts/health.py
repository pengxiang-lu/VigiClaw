#!/usr/bin/env python3
"""
Robot Health Check — 检查 Server 健康状态

用法:
    python health.py

输出:
    Server 是否正常运行

示例:
    python health.py
"""

import requests
import sys

SERVER_URL = "http://127.0.0.1:8000"


def check_health():
    """健康检查"""
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code == 200:
            print("OK: Server 正常运行")
            print(f"响应: {resp.text}")
        else:
            print(f"错误 (HTTP {resp.status_code}): {resp.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到 Server")
        print("提示: 请确认 server.py 正在运行")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    check_health()
