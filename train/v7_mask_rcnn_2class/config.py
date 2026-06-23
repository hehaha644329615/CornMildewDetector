# 复用原有v1、v2路径、基础超参，新增Mask专属
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
TRAIN_ROOT = os.path.join(cur_path, "..")
COMMON_ROOT = os.path.join(TRAIN_ROOT, "common_2cls")

# 数据集
VOC_ROOT = os.path.join(COMMON_ROOT, "data", "VOC2012")
JSON_CLASS_PATH = os.path.join(COMMON_ROOT, "data","VOC2012", "pascal_voc_classes.json")
PRETRAIN_FASTER = os.path.join(COMMON_ROOT, "fasterrcnn_resnet50_fpn_coco.pth")
PRETRAIN_MASK = os.path.join(COMMON_ROOT, "maskrcnn_resnet50_fpn_coco.pth")

# 训练基础超参
NUM_CLASSES = 2
EPOCHS = 15
BATCH_SIZE = 4
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
STEP_LR_GAMMA = 0.33
STEP_SIZE = 3
ASPECT_RATIO_GROUP = 3
TRAINABLE_LAYERS = 3

# MaskRCNN专属
MASK_POOL_SIZE = 14
MASK_LOSS_WEIGHT = 1.0

# 推理阈值
BOX_THRESHOLD = 0.5

# 输出路径
SAVE_WEIGHT_DIR = os.path.join(cur_path, "save_weights")
DEVICE = "cuda:0"
USE_AMP = False

# 绘图字体
FONT_MAC = "/System/Library/Fonts/PingFang.ttc"
FONT_WIN = "arial.ttf"
LINE_THICKNESS = 3
FONT_SIZE = 20