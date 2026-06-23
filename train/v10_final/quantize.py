"""CoreML INT8 量化"""
from ultralytics import YOLO
model = YOLO('v10_final/production/weights/best.pt')
model.export(format='coreml', imgsz=640, int8=True, nms=True)
print("✅ CoreML INT8 导出完成")
