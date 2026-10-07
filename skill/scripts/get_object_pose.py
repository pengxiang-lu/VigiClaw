#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import os
import requests
import json
import sys
import cv2
from datetime import datetime
SERVER_URL = "http://127.0.0.1:8000"


def get_depth_from_image(depth_image, u, v):
    """从深度图像中获取指定像素位置的深度值
    
    深度图是 16 位编码，最大深度 100m
    需要将 16 位值转换为实际的深度值（米）
    """
    try:
        depth_array = np.array(depth_image)
        
        # 根据数组维度处理不同格式的深度图像
        if len(depth_array.shape) == 4:
            # 形状如 (1, 128, 128, 1) - 去掉 batch 和 channel 维度
            depth_array = depth_array[0, :, :, 0]
        elif len(depth_array.shape) == 3:
            # 如果是 (H, W, 1) 或 (1, H, W) 格式
            if depth_array.shape[2] == 1:
                depth_array = depth_array[:, :, 0]
            elif depth_array.shape[0] == 1:
                depth_array = depth_array[0]
        
        # 确保是 2D 数组
        if len(depth_array.shape) != 2:
            # 尝试 squeeze 去掉所有长度为 1 的维度
            depth_array = np.squeeze(depth_array)
            if len(depth_array.shape) != 2:
                print(f"错误：无法将深度图像转换为 2D 数组，当前形状：{depth_array.shape}")
                return None
        
        # 检查坐标是否在图像范围内
        if v < 0 or v >= depth_array.shape[0] or u < 0 or u >= depth_array.shape[1]:
            print(f"错误：像素坐标 ({u}, {v}) 超出图像范围 {depth_array.shape}")
            return None
        
        # 获取 16 位深度值
        depth = float(depth_array[v, u])/1000
        print(f"深度值（米）：{depth}")
        return depth
        
    except Exception as e:
        print(f"错误：无法读取深度值 - {e}")
        return None


def camera_to_world(u, v, depth, intrinsic, extrinsic):
    """
    将相机坐标系中的像素坐标转换为世界坐标系
    
    参数:
        u, v: 像素坐标（列，行）
        depth: 深度值（米）
        intrinsic: 3x3 相机内参矩阵
        extrinsic: 4x4 相机外参矩阵（相机→世界变换）
    
    返回:
        [x, y, z] 世界坐标系中的位置（米）
    """
    # 将输入的内参和外参转换为numpy数组
    intrinsic = np.array(intrinsic)
    extrinsic = np.array(extrinsic)
    
    # 正确处理数组维度
    if len(intrinsic.shape) == 3:
        intrinsic = intrinsic[0]
    elif len(intrinsic.shape) == 1:
        # 如果是一维数组，需要重塑为3x3矩阵
        intrinsic = intrinsic.reshape(3, 3)
    
    if len(extrinsic.shape) == 3:
        extrinsic = extrinsic[0]
    elif extrinsic.shape == (3, 4):
        # 如果是3x4矩阵，扩展为4x4齐次变换矩阵
        extrinsic_homogeneous = np.eye(4)
        extrinsic_homogeneous[:3, :4] = extrinsic
        extrinsic = extrinsic_homogeneous
    elif len(extrinsic.shape) == 1:
        # 如果是一维数组，需要重塑为4x4矩阵
        extrinsic = extrinsic.reshape(4, 4)
    
    # 计算外参矩阵的逆矩阵（用于从相机坐标系转换到世界坐标系）
    extrinsic_inv = np.linalg.inv(extrinsic)
    
    # 从像素坐标(u,v)和深度值计算相机坐标系中的3D坐标
    # 相机内参矩阵的参数提取
    fx = intrinsic[0, 0]  # x轴焦距
    fy = intrinsic[1, 1]  # y轴焦距
    cx = intrinsic[0, 2]  # x轴主点偏移
    cy = intrinsic[1, 2]  # y轴主点偏移
    
    # 使用针孔相机模型公式将像素坐标转换为相机坐标系下的3D坐标
    x_cam = (u - cx) * depth / fx  # 相机坐标系下的x坐标
    y_cam = (v - cy) * depth / fy  # 相机坐标系下的y坐标
    z_cam = depth                  # 相机坐标系下的z坐标（即深度值）
    
    # 构造齐次坐标（4维向量）
    point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    
    # 使用外参矩阵的逆矩阵将相机坐标系下的点转换到世界坐标系
    point_world = extrinsic_inv @ point_cam  # 矩阵乘法
    
    # 如果齐次坐标的w分量不为0，则进行归一化（除以w分量）
    if point_world[3] != 0:
        point_world = point_world[:3] / point_world[3]  # 取前3个分量并除以w分量
    else:
        # 如果w分量为0，直接取前3个分量
        point_world = point_world[:3]
    
    # 临时调试代码（已注释）：可能用于调整坐标系方向
    # temp = point_world[0]
    # point_world[0] = -point_world[1]
    # point_world[1] = temp

    # 返回世界坐标系下的3D坐标（列表形式）
    return point_world.tolist()


def euler_to_rotation_matrix(roll, pitch, yaw):
    """欧拉角 Z-Y-X (rad) -> 3x3 旋转矩阵"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return R


def rotation_matrix_to_euler(R):
    """3x3 旋转矩阵 -> Z-Y-X 欧拉角 (roll, pitch, yaw) rad"""
    sy = -R[2, 0]
    pitch = np.arcsin(np.clip(sy, -1.0, 1.0))
    cp = np.cos(pitch)
    if np.abs(cp) > 1e-6:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = 0.0
        roll = np.arctan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def rotate_pose_yaw_clockwise_90(pose):
    """将位姿中的 yaw 顺时针旋转 90 度（弧度）。"""
    if len(pose) < 6:
        return pose

    adjusted_pose = list(pose)
    # adjusted_pose[5] = float(adjusted_pose[5] + np.pi / 2)
    return adjusted_pose


def pose_to_homogeneous_matrix(pose):
    """[x,y,z,roll,pitch,yaw] -> 4x4 齐次矩阵"""
    x, y, z = pose[0], pose[1], pose[2]
    roll, pitch, yaw = pose[3], pose[4], pose[5]
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def homogeneous_to_pose(T):
    """4x4 齐次矩阵 -> [x,y,z,roll,pitch,yaw] (m, rad)"""
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    R = T[:3, :3]
    roll, pitch, yaw = rotation_matrix_to_euler(R)
    return [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """四元数 (x, y, z, w) -> 3x3 旋转矩阵"""
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float32,
    )


def rotation_matrix_to_quaternion(R):
    """3x3 旋转矩阵 -> 四元数 (qx, qy, qz, qw)"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return (float(qx), float(qy), float(qz), float(qw))


def compute_object_pose_from_pcd(pts_base):
    """
    从3D点云计算物体姿态
    使用PCA分析确定主要方向，然后构建一个表示物体方向的旋转矩阵
    输入点云应为滤波后、基座系下的点云；返回 (center, R_pose) 或 None。
    """
    pts = np.asarray(pts_base, dtype=np.float64)
    if pts.shape[0] < 4:
        return None
    
    # 1) 中心点（对应 pcl::compute3DCentroid）
    pca_centroid = np.mean(pts, axis=0)
    
    # 2) 协方差矩阵（PCL 归一化：除以 n-1）
    centered = pts - pca_centroid
    covariance = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    
    # 3) 特征值、特征向量（对应 Eigen::SelfAdjointEigenSolver）
    eigen_values, eigen_vectors = np.linalg.eigh(covariance)
    
    # 4) 确保特征向量构成右手坐标系
    ev = eigen_vectors
    ev = ev.copy()
    ev[:, 2] = np.cross(ev[:, 0], ev[:, 1])
    ev[:, 1] = np.cross(ev[:, 2], ev[:, 0])
    ev[:, 0] = np.cross(ev[:, 1], ev[:, 2])
    for i in range(3):
        n = np.linalg.norm(ev[:, i])
        if n > 1e-10:
            ev[:, i] = ev[:, i] / n
    
    # 5) 按特征值降序排列特征向量
    indices = np.argsort(eigen_values)[::-1]
    sorted_eigen_vectors = ev[:, indices].copy()
    
    # 6) 确保右手系（行列式为正）
    if np.linalg.det(sorted_eigen_vectors) < 0:
        sorted_eigen_vectors[:, 2] = -sorted_eigen_vectors[:, 2]
    
    # 返回质心和旋转矩阵
    center = np.array(pca_centroid, dtype=np.float32)
    R_pose = np.array(sorted_eigen_vectors, dtype=np.float32)
    
    return center, R_pose


def detect_and_segment_objects():
    """从服务器获取RGB图像和深度图像，使用YOLO分割模型检测物体并计算位姿"""
    try:
        state_resp = requests.get(f"{SERVER_URL}/state", timeout=10)
        if state_resp.status_code != 200:
            print(f"错误：无法获取机器人状态。HTTP 状态码：{state_resp.status_code}")
            sys.exit(1)
        
        state = state_resp.json()
        
        rgb_image_data = state["base_camera_rgb"]
        rgb_array = np.array(rgb_image_data)
        
        if rgb_array.ndim == 4:
            rgb_array = rgb_array[0]
        elif rgb_array.ndim == 3 and rgb_array.shape[0] == 3:
            rgb_array = np.transpose(rgb_array, (1, 2, 0))
        
        if rgb_array.dtype == np.float32 or rgb_array.dtype == np.float64:
            if rgb_array.max() <= 1.0:
                rgb_array = (rgb_array * 255).astype(np.uint8)
            else:
                rgb_array = rgb_array.astype(np.uint8)
        else:
            rgb_array = rgb_array.astype(np.uint8)
        
        rgb_image = Image.fromarray(rgb_array)
        
        rgb_dir = "rgb_images"
        os.makedirs(rgb_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_png_path = os.path.join(rgb_dir, f"temp_input_{timestamp}.png")
        
        img_uint8 = rgb_array.astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_png_path, img_bgr)
        
        model = YOLO("yolo_seg.pt")
        
        results = model(
            source=temp_png_path,
            imgsz=512,
            conf=0.4,
            iou=0.5,
            agnostic_nms=False,
            max_det=100,
            save=False,
            show_labels=True,
            show_conf=True,
            verbose=False
        )
        
        all_objects_info = []
        
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            
            for i, mask in enumerate(masks):
                cls_id = int(boxes.cls[i])
                cls_name = model.names[cls_id]
                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask[0]
                
                all_objects_info.append({
                    'u': center_x,
                    'v': center_y,
                    'mask': mask,
                    'score': conf,
                    'description': cls_name,
                    'bbox': [x1, y1, x2, y2]
                })
        
        return all_objects_info, rgb_image, state
        
    except requests.exceptions.ConnectionError:
        print("错误：无法连接到服务器")
        print("提示：请确认 server.py 正在运行")
        sys.exit(1)
    except Exception as e:
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_object_pointcloud(mask, depth_image, intrinsic, extrinsic):
    """根据分割掩码和深度图像生成物体的3D点云"""
    try:
        # 深度图处理
        depth_array = np.array(depth_image)
        depth_array = depth_array[0, :, :, 0]  # 移除批次和通道维度
        
        if len(depth_array.shape) != 2:
            depth_array = np.squeeze(depth_array)
            if len(depth_array.shape) != 2:
                print(f"错误：无法将深度图像转换为 2D 数组，当前形状：{depth_array.shape}")
                return None
        
        # 确保掩码和深度图尺寸匹配
        if mask.shape != depth_array.shape:
            mask_resized = np.zeros_like(depth_array)
            h, w = depth_array.shape
            mask_resized[:mask.shape[0], :mask.shape[1]] = mask
            mask = mask_resized
        
        # 获取有效的深度点（在掩码内的点）
        valid_depth_mask = (depth_array > 0) & mask
        y_coords, x_coords = np.where(valid_depth_mask)
        
        if len(x_coords) == 0:
            print("错误：没有有效的深度点")
            return None
        
        # 获取对应的深度值
        depths = depth_array[y_coords, x_coords] / 1000.0  # 转换为米
        
        # 相机内参
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        
        # 将像素坐标转换为相机坐标系
        x_cam = (x_coords - cx) * depths / fx
        y_cam = (y_coords - cy) * depths / fy
        z_cam = depths
        
        # 构造相机坐标系下的点
        points_cam = np.vstack((x_cam, y_cam, z_cam)).T
        
        # 确保外参矩阵是4x4的齐次变换矩阵
        if len(extrinsic.shape) == 1:
            # 如果是一维数组，重塑为4x4矩阵
            extrinsic = extrinsic.reshape(4, 4)
        elif extrinsic.shape == (3, 4):
            # 如果是3x4矩阵，扩展为4x4齐次变换矩阵
            extrinsic_homogeneous = np.eye(4)
            extrinsic_homogeneous[:3, :4] = extrinsic
            extrinsic = extrinsic_homogeneous
        
        # 转换到世界坐标系
        extrinsic_inv = np.linalg.inv(extrinsic)
        
        # 将3D点转换到世界坐标系
        points_world = []
        for point in points_cam:
            point_cam_h = np.array([point[0], point[1], point[2], 1.0])
            point_world_h = extrinsic_inv @ point_cam_h
            if point_world_h[3] != 0:
                point_world = point_world_h[:3] / point_world_h[3]
            else:
                point_world = point_world_h[:3]
            points_world.append(point_world)
        
        if len(points_world) == 0:
            print("错误：转换后没有有效的3D点")
            return None
        
        return np.array(points_world)
    
    except Exception as e:
        print(f"错误：生成点云时出现问题 - {e}")
        import traceback
        traceback.print_exc()
        return None


def save_marked_image_with_cross(rgb_image, objects_info):
    """在图像上添加多个物体的十字标记并保存标记后的RGB图像，支持根据yaw角度旋转十字标记"""
    try:
        # 转换为可绘制的图像
        draw = ImageDraw.Draw(rgb_image)
        
        # 定义不同颜色用于区分不同物体
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 青色
            (255, 128, 0),  # 橙色
            (128, 0, 255),  # 紫色
        ]
        
        for idx, obj in enumerate(objects_info):
            u = obj['u']
            v = obj['v']
            yaw = obj.get('yaw', None)
            
            # 选择颜色（循环使用）
            color = colors[idx % len(colors)]
            
            if yaw is not None:
                # 根据yaw角度旋转十字标记
                import math
                
                # 定义十字的端点（相对于中心点）
                half_length = 10  # 十字长度的一半
                
                # 计算旋转前的四个端点
                # 水平线的两个端点
                p1_orig = (-15, 0)  # 左端点
                p2_orig = (15, 0)   # 右端点
                # 垂直线的两个端点
                p3_orig = (0, -8)  # 上端点
                p4_orig = (0, 8)   # 下端点
                
                # 应用旋转矩阵
                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)
                
                # 旋转后的端点坐标
                p1_rot = (u + p1_orig[0] * cos_yaw - p1_orig[1] * sin_yaw,
                          v + p1_orig[0] * sin_yaw + p1_orig[1] * cos_yaw)
                p2_rot = (u + p2_orig[0] * cos_yaw - p2_orig[1] * sin_yaw,
                          v + p2_orig[0] * sin_yaw + p2_orig[1] * cos_yaw)
                p3_rot = (u + p3_orig[0] * cos_yaw - p3_orig[1] * sin_yaw,
                          v + p3_orig[0] * sin_yaw + p3_orig[1] * cos_yaw)
                p4_rot = (u + p4_orig[0] * cos_yaw - p4_orig[1] * sin_yaw,
                          v + p4_orig[0] * sin_yaw + p4_orig[1] * cos_yaw)
                
                # 绘制旋转后的十字
                draw.line([p1_rot, p2_rot], fill=color, width=2)  # 旋转后的水平线
                draw.line([p3_rot, p4_rot], fill=color, width=2)  # 旋转后的垂直线
            else:
                # 绘制标准的水平/垂直十字
                # 绘制水平线
                draw.line([(u - 15, v), (u + 15, v)], fill=color, width=2)
                # 绘制垂直线
                draw.line([(u, v - 8), (u, v + 8)], fill=color, width=2)
            
            # 在十字旁边添加物体编号
            draw.text((u + 12, v - 12), str(idx + 1), fill=color)
        
        # 获取当前时间戳用于文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存带标记的RGB图像
        output_dir = os.path.join(os.path.dirname(__file__), "detected_images")
        os.makedirs(output_dir, exist_ok=True)
        marked_rgb_filename = os.path.join(
            output_dir, f"base_camera_rgb_marked_{timestamp}.png"
        )
        rgb_image.save(marked_rgb_filename)
        print(f"带标记的RGB图像已保存至: {marked_rgb_filename}")
        
        return marked_rgb_filename
        
    except Exception as e:
        print(f"错误：保存图像时出现问题 - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def convert(u, v, state, mask=None, object_index=None, object_name=None):
    """使用提供的state数据进行坐标转换，如果有mask则同时计算物体姿态
    
    Args:
        u: 图像x坐标
        v: 图像y坐标
        state: 机器人状态
        mask: 物体分割掩码（可选）
        object_index: 物体索引（可选）
        object_name: 物体名称/描述（可选）
    """
    try:
        # 从传入的状态获取相机参数
        params_resp = requests.get(f"{SERVER_URL}/camera_params", timeout=10)
        if params_resp.status_code != 200:
            print(f"错误：无法获取相机参数。HTTP 状态码：{params_resp.status_code}")
            sys.exit(1)
        
        params = params_resp.json()
        
        # 提取深度图像
        depth_image = state["base_camera_depth"]
        
        # 提取相机参数
        intrinsic = np.array(params["base_intrinsic_cv"])
        extrinsic = np.array(params["base_extrinsic_cv"])
        
        # 正确处理数组维度
        if len(intrinsic.shape) == 3:
            intrinsic = intrinsic[0]
        elif len(intrinsic.shape) == 1:
            # 如果是一维数组，需要重塑为3x3矩阵
            intrinsic = intrinsic.reshape(3, 3)
        
        if len(extrinsic.shape) == 3:
            extrinsic = extrinsic[0]
        elif extrinsic.shape == (3, 4):
            # 如果是3x4矩阵，扩展为4x4齐次变换矩阵
            extrinsic_homogeneous = np.eye(4)
            extrinsic_homogeneous[:3, :4] = extrinsic
            extrinsic = extrinsic_homogeneous
        elif len(extrinsic.shape) == 1:
            # 如果是一维数组，需要重塑为4x4矩阵
            extrinsic = extrinsic.reshape(4, 4)
        
        # 如果提供了掩码，计算完整位姿
        if mask is not None:
            # 生成物体的3D点云
            object_pcd = generate_object_pointcloud(mask, depth_image, intrinsic, extrinsic)
            
            if object_pcd is not None and len(object_pcd) > 0:
                # 使用点云的质心作为物体位置
                object_center = np.mean(object_pcd, axis=0)
                
                # 计算物体的姿态
                pose_result = compute_object_pose_from_pcd(object_pcd)
                
                if pose_result is not None:
                    center, rotation_matrix = pose_result
                    
                    # 将旋转矩阵转换为欧拉角 (Z-Y-X, 弧度)
                    roll, pitch, yaw = rotation_matrix_to_euler(rotation_matrix)
                    yaw = -yaw
                    # 构造完整位姿 [x, y, z, roll, pitch, yaw]
                    full_pose = [float(center[0]), float(center[1]), float(center[2]), 
                                 float(roll), float(pitch), float(yaw)]
                    
                    if object_name is not None:
                        pass
                    elif object_index is not None:
                        pass
                    else:
                        pass
                    return rotate_pose_yaw_clockwise_90(full_pose)
                else:
                    # 如果无法计算姿态，只返回位置信息，z坐标除以2
                    pose_only_position = [float(object_center[0]), float(object_center[1]), float(object_center[2]), 0, 0, 0]
                    if object_name is not None:
                        pass
                    elif object_index is not None:
                        pass
                    else:
                        pass
                    return pose_only_position
            else:
                print("无法生成物体点云")
        
        # 如果没有掩码或者无法计算完整位姿，使用原来的单点方法
        depth = get_depth_from_image(depth_image, u, v)
        
        if depth <= 0:
            print(f"错误：无效的深度值 {depth}（必须 > 0）")
            sys.exit(1)
        
        # 进行坐标转换
        world_coords = camera_to_world(u, v, depth, intrinsic, extrinsic)
        
        # 对于单点，我们只能返回位置信息，姿态设为零欧拉角
        pose_with_unit_orientation = world_coords + [0, 0, 0]  # [x, y, z, roll, pitch, yaw]
        if object_index is not None:
            pass
        else:
            pass
        
        return pose_with_unit_orientation
        
    except requests.exceptions.ConnectionError:
        print("错误：无法连接到服务器")
        print("提示：请确认 server.py 正在运行")
        sys.exit(1)
    except Exception as e:
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    print("正在处理图像，稍等片刻...")
    
    result = detect_and_segment_objects()
    
    if result is None:
        print("[None]")
        return

    objects_info, rgb_image, state = result
    
    if len(objects_info) == 0:
        print("未检测到任何物体")
        return
    
    world_poses = []
    for idx, obj in enumerate(objects_info):
        u = obj['u']
        v = obj['v']
        mask = obj['mask']
        description = obj.get('description', f'物体{idx+1}')
        
        u_int = int(round(u))
        v_int = int(round(v))
        
        world_pose = convert(u_int, v_int, state, mask, object_index=idx+1, object_name=description)
        
        if len(world_pose) >= 6:
            obj['yaw'] = world_pose[5]
        else:
            obj['yaw'] = None
        
        world_poses.append(world_pose)
    
    save_marked_image_with_cross(rgb_image, objects_info)
    
    print("\n检测到物体及其位姿(x, y, z, roll, pitch, yaw)如下：")
    for idx, (obj, pose) in enumerate(zip(objects_info, world_poses), 1):
        print(f"物体{idx}：{obj['description']}，位姿：[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}, {pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}]")
  

if __name__ == "__main__":
    main()
