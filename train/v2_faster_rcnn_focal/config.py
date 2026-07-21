import os

# ===================== 1. 项目根路径配置（和train路径逻辑统一） =====================
cur_path = os.path.dirname(os.path.abspath(__file__))
TRAIN_ROOT = os.path.join(cur_path, "..")
COMMON_ROOT = os.path.join(TRAIN_ROOT, "common_3cls")

# 数据集路径
VOC_ROOT = os.path.join(COMMON_ROOT, "data", "VOC2012")
ANNOTATION_PATH = os.path.join(VOC_ROOT, "Annotations")
# 双层JPEGImages软链接图片目录
IMG_ROOT = os.path.join(VOC_ROOT, "JPEGImages", "JPEGImages")
JSON_CLASS_PATH = os.path.join(COMMON_ROOT, "data", "pascal_voc_classes.json")

# 预训练COCO权重路径
PRETRAIN_WEIGHT = os.path.join(COMMON_ROOT, "fasterrcnn_resnet50_fpn_coco.pth")

# 训练输出路径
SAVE_WEIGHT_DIR = os.path.join(cur_path, "save_weights")
LOG_SAVE_PATH = cur_path  # results日志、曲线图片保存在v1_faster_rcnn根目录

# ===================== 2. 训练超参配置 =====================
# 数据集类别：不含背景，你玉米3分类
NUM_CLASSES = 3
EPOCHS = 15
BATCH_SIZE = 4
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
STEP_LR_GAMMA = 0.33
STEP_SIZE = 3
ASPECT_RATIO_GROUP = 3

# 模型配置
TRAINABLE_LAYERS = 3  # backbone解冻层数
RPN_SCORE_THRESH = 0.5

# 推理阈值
BOX_THRESHOLD = 0.5

# 设备
DEVICE = "cuda:0"
USE_AMP = False

# ===================== 3. 绘图/可视化配置 =====================
LINE_THICKNESS = 3
FONT_SIZE = 20
# Mac兼容字体
FONT_MAC = "/System/Library/Fonts/PingFang.ttc"
FONT_WIN = "arial.ttf"

# Focal Loss 超参（v2独有）
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0