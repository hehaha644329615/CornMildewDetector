# v6 YOLOv8 优化配置

MODEL_NAME = "yolov8n.pt"
DATA_YAML = "../common_3cls/data/data.yaml"
EPOCHS = 80
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = "cuda:0"
PROJECT = "v6_yolov8"
EXPERIMENT = "optimized"

# 优化参数（v5 跑完后根据结果调整）
LR0 = 0.001
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3
