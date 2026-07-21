"""导出 ONNX 并量化"""
from ultralytics import YOLO
model = YOLO('v10_final/production/weights/best.pt')
model.export(format='onnx', imgsz=640, int8=True)
print("✅ ONNX INT8 导出完成")
