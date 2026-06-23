"""
v6 YOLOv8 优化训练
"""
from ultralytics import YOLO
import torch
import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.join(cur_path, "..")
sys.path.append(train_root)

DATA_YAML = os.path.join(train_root, "common_3cls", "data", "YOLO_data", "data.yaml")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")
print(f"数据路径: {DATA_YAML}")

model = YOLO('yolov8n.pt')
model.train(
    data=DATA_YAML,
    epochs=80, imgsz=640, batch=16,
    device=device, project='v6_yolov8', name='optimized'
)