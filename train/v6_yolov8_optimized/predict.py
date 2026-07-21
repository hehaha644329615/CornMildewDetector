"""
v6 YOLOv8 预测脚本
"""
from ultralytics import YOLO

model = YOLO('v6_yolov8/optimized/weights/best.pt')
model.predict(source='test.jpg', save=True, conf=0.5)
