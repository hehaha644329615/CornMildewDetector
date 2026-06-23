"""
v5 YOLOv8 预测脚本
"""
from ultralytics import YOLO

model = YOLO('v5_yolov8/baseline/weights/best.pt')
model.predict(source='test.jpg', save=True, conf=0.5)
