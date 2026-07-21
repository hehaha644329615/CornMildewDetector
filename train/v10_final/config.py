# v10 最终上线版配置
import os
cur_path = os.path.dirname(os.path.abspath(__file__))

NUM_CLASSES = 3
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = "cuda:0"
QUANTIZATION = "INT8"
DEPLOY_TARGET = "iOS CoreML"
