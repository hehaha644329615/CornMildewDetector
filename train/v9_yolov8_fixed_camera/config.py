# v9 固定相机方案配置
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
TRAIN_ROOT = os.path.join(cur_path, "..")

NUM_CLASSES = 3
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = "cuda:0"
# 固定采集参数
HARDWARE_FRAME = "60×60cm"
CAMERA_DISTANCE = "fixed"
