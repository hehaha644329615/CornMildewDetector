"""
v5 YOLOv8 评估脚本
"""
from ultralytics import YOLO

model = YOLO('v5_yolov8/baseline/weights/best.pt')
model.val()
