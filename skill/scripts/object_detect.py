import requests
import sys
import numpy as np
from ultralytics import YOLO
import torch
import cv2
import os
from datetime import datetime
SERVER_URL = "http://127.0.0.1:8000"

def calculate_iou(box1, box2):
    """计算两个边界框的IoU（交并比）"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    # 计算各自面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def nms_filter(detections, iou_threshold=0.5, class_agnostic=True):
    """
    手动实现NMS过滤
    
    参数:
    - detections: 检测结果列表，每个元素是字典，包含'bbox', 'conf', 'class'
    - iou_threshold: IoU阈值
    - class_agnostic: 是否使用类别无关的NMS
    """
    if not detections:
        return []
    
    # 按置信度排序
    detections.sort(key=lambda x: x['conf'], reverse=True)
    
    keep = []
    while detections:
        # 取出置信度最高的检测
        best = detections.pop(0)
        keep.append(best)
        
        to_remove = []
        for i, det in enumerate(detections):
            # 如果启用类别无关NMS，或者类别相同，则计算IoU
            if class_agnostic or det['class'] == best['class']:
                iou = calculate_iou(best['bbox'], det['bbox'])
                if iou > iou_threshold:
                    to_remove.append(i)
        
        # 从后往前移除，避免索引变化
        for idx in reversed(to_remove):
            detections.pop(idx)
    
    return keep

def get_rgb_image():
    """从机器人服务器获取 RGB 图像"""
    resp = requests.get(f"{SERVER_URL}/state", timeout=10)
    if resp.status_code != 200:
        print(f"错误 (HTTP {resp.status_code}): {resp.text}")
        sys.exit(1)
    
    state = resp.json()
    
    # 获取RGB图像数据
    rgb_image_data = state["base_camera_rgb"]
    
    # 调试：打印数据形状和类型
    # print(f"原始RGB数据形状: {np.array(rgb_image_data).shape}")
    # print(f"原始RGB数据类型: {type(rgb_image_data)}")
    
    rgb_array = np.array(rgb_image_data)
    
    # 检查数组维度并进行相应处理
    if rgb_array.ndim == 4:  # 如果是 (batch, height, width, channels) 格式
        rgb_array = rgb_array[0]  # 移除batch维度
    elif rgb_array.ndim == 3 and rgb_array.shape[0] == 3:  # 如果是 (channels, height, width) 格式
        rgb_array = np.transpose(rgb_array, (1, 2, 0))  # 转换为 (height, width, channels)
    
    
    return rgb_array

def main():
    # 加载分割模型
    model = YOLO("yolo_seg.pt")
    
    # 从机器人获取图像
    img = get_rgb_image()
    
    # 将图像保存为PNG文件
    rgb_dir = "rgb_images"
    os.makedirs(rgb_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_png_path = os.path.join(rgb_dir, f"temp_input_{timestamp}.png")
    
    # 先转换数据类型为uint8，再转换为BGR格式
    img_uint8 = img.astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_png_path, img_bgr)
    

    # 使用PNG文件进行YOLO分割
    results = model(
        source=temp_png_path,
        imgsz=512,
        conf=0.4,          # 置信度阈值
        iou=0.5,           # NMS IoU阈值
        agnostic_nms=False,
        max_det=100,       # 增加最大检测数
        save=False,
        show_labels=True,
        show_conf=True,
        verbose=False
    )
    
    # 保存分割结果图像
    output_dir = "detected_images"
    os.makedirs(output_dir, exist_ok=True)
    
    if results[0].orig_img is not None:
        result_img = results[0].plot()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"segmentation_{timestamp}.jpg")
        cv2.imwrite(output_path, result_img)
        print(f"\n分割结果已保存到: {output_path}")
    
    # 收集分割结果
    segmentations = []
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes
        
        for i, mask in enumerate(masks):
            cls_id = int(boxes.cls[i])
            cls_name = model.names[cls_id]
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            
            # 计算掩码面积
            mask_area = int(np.sum(mask > 0.5))
            
            segmentations.append({
                'class': cls_name,
                'conf': conf,
                'bbox': [x1, y1, x2, y2],
                'mask': mask,
                'mask_area': mask_area
            })
    
    # 打印分割结果
    print("\n检测到物体如下：")
    for idx, seg in enumerate(segmentations, 1):
        x1, y1, x2, y2 = seg['bbox']
        print(f"物体{idx}：{seg['class']}，位置：({x1},{y1})-({x2},{y2})")
        
    return segmentations

if __name__ == "__main__":
    # 设置numpy打印选项
    np.set_printoptions(precision=3, suppress=True)
    
    # 运行主函数
    detections = main()