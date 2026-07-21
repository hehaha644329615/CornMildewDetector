"""
v9 YOLOv8 固定相机方案
关键突破：统一拍照距离和角度（60×60cm簸箕 + 手机支架）
"""
import os, sys, torch
from ultralytics import YOLO

cur_path = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.join(cur_path, "..")
sys.path.append(train_root)

DATA_YAML = os.path.join(train_root, "common_3cls", "data", "YOLO_data", "data.yaml")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = YOLO('yolov8n.pt')
model.train(data=DATA_YAML, epochs=50, imgsz=640, batch=16,
            device=device, project=os.path.join(cur_path, 'v9_yolov8'), name='fixed_camera')
