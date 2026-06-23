"""
v6 YOLOv8 评估脚本
"""
from ultralytics import YOLO

model = YOLO('v6_yolov8/optimized/weights/best.pt')
model.val()
