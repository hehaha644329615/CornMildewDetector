"""
v1 Faster RCNN 验证集 GT vs 预测框 可视化
"""
import os
import sys
import random
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

cur_path = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.join(cur_path, "..")
sys.path.append(train_root)

# 解决matplotlib中文乱码
plt.rcParams["font.sans-serif"] = ["Songti SC"]
plt.rcParams["axes.unicode_minus"] = False


import config as cfg
from common_3cls.FasterRCNN_models.network_files import FasterRCNN
from common_3cls.FasterRCNN_models.backbone import resnet50_fpn_backbone
from torchvision import transforms

# ========== 配置 ==========
VAL_ROOT = os.path.join(cfg.VOC_ROOT)
IMG_DIR = os.path.join(VAL_ROOT, "JPEGImages")
ANN_DIR = os.path.join(VAL_ROOT, "Annotations")
VAL_TXT = os.path.join(VAL_ROOT, "ImageSets", "Main", "val.txt")
WEIGHT_PATH = os.path.join(cur_path, "save_weights", "resNetFpn-model-14.pth")
NUM_CLASSES = cfg.NUM_CLASSES + 1
CONF_THRESH = 0.45

# 类别颜色
CLASS_COLORS = {1: 'green', 2: 'orange', 3: 'red'}
CLASS_NAMES = {1: 'healthy', 2: 'light_mold', 3: 'heavy_mold'}

# ========== 加载模型 ==========
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
model = FasterRCNN(backbone, num_classes=NUM_CLASSES)

if os.path.exists(WEIGHT_PATH):
    ckpt = torch.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print("✅ 权重加载成功")
else:
    print("⚠️ 未找到权重文件，只画真实框")
model.to(device)
model.eval()

# ========== 读取验证集图片列表 ==========
with open(VAL_TXT, 'r') as f:
    val_ids = [line.strip() for line in f if line.strip()]

# 随机选 2 张
selected = random.sample(val_ids, min(2, len(val_ids)))

fig, axes = plt.subplots(1, 2, figsize=(30, 20))

for ax, img_id in zip(axes.flat, selected):
    # 读图片
    img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(IMG_DIR, f"{img_id}.png")
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # ---------- 画 GT 框（绿色） ----------
    ann_path = os.path.join(ANN_DIR, f"{img_id}.xml")
    if os.path.exists(ann_path):
        import xml.etree.ElementTree as ET
        tree = ET.parse(ann_path)
        for obj in tree.findall("object"):
            name = obj.find("name").text
            bnd = obj.find("bndbox")
            x1, y1 = float(bnd.find("xmin").text), float(bnd.find("ymin").text)
            x2, y2 = float(bnd.find("xmax").text), float(bnd.find("ymax").text)
            cls_id = {"healthy": 1, "light_mold": 2, "heavy_mold": 3}.get(name, 1)
            draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
            draw.text((x1, y1-10), name, fill='green')
    
    # ---------- 画预测框（红色） ----------
    if os.path.exists(WEIGHT_PATH):
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor)[0]
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score >= CONF_THRESH:
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline='red', width=10)
                draw.text((x1, y2+2), f"{CLASS_NAMES.get(label,'?')}:{score:.2f}", fill='red')
    
    ax.imshow(img)
    ax.set_title(f"{img_id}\n绿框=GT  红框=Pred", fontsize=9)
    ax.axis('off')

plt.suptitle("v1 Faster RCNN 验证集 GT vs 预测框", fontsize=24, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(cur_path, "gt_vs_pred.jpg")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✅ 保存至 {save_path}")