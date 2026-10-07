#!/usr/bin/env python3
"""
Robot Get RGB Image — 获取机器人摄像头图像

用法:
    python get_rgb_image.py

输出:
    自动保存底座摄像头图像为 PNG 文件

示例:
    python get_rgb_image.py
"""

import requests
import sys
import json
import argparse
import os
from datetime import datetime
import numpy as np

try:
    from PIL import Image
except ImportError:
    print("提示：安装 PIL 以支持图像保存功能：pip install Pillow")
    Image = None

SERVER_URL = "http://127.0.0.1:8000"


def save_image(img_array, output_path):
    """将图像数组保存为 PNG 文件"""
    if Image is None:
        print("警告：PIL 未安装，无法保存图像")
        return False
    
    try:
        # 转换形状：从 (1, H, W, 3) 到 (H, W, 3)
        img_data = np.array(img_array).astype(np.uint8)
        if img_data.ndim == 4 and img_data.shape[0] == 1:
            img_data = img_data[0]
        
        # 创建 PIL 图像并保存
        img = Image.fromarray(img_data, mode='RGB')
        img.save(output_path, format='PNG')
        print(f"✓ 图像已保存：{output_path}")
        return True
    except Exception as e:
        print(f"✗ 保存图像失败：{e}")
        return False


def get_state():
    """获取机器人摄像头图像"""
    resp = requests.get(f"{SERVER_URL}/state", timeout=10)

    if resp.status_code != 200:
        print(f"错误 (HTTP {resp.status_code}): {resp.text}")
        sys.exit(1)

    state = resp.json()

    # 获取并保存摄像头图像
    cam = "base_camera_rgb"
    img = state.get(cam)
    if img and isinstance(img, list):
        shape = np.array(img).shape
        print(f"{cam}: {shape}")
        
        # 默认保存图像
        if Image is not None:
            output_dir = os.path.join(os.path.dirname(__file__), "rgb_images")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{cam}_{timestamp}.png"
            output_path = os.path.join(output_dir, filename)
            save_image(img, output_path)

    return state


if __name__ == "__main__":
    state = get_state()
