import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET
import argparse
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.join(cur_path, "..")
sys.path.append(train_root)

import config as cfg
from common_2cls.MaskRCNN_models.network_files import MaskRCNN
from common_2cls.MaskRCNN_models.backbone import resnet50_fpn_backbone

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 命令行动态传参 ======================
parser = argparse.ArgumentParser(description="Mask RCNN 评估脚本")
parser.add_argument("--conf", type=float, default=0.5, help="检测框置信过滤阈值 CONF_THRESH")
parser.add_argument("--iou_match", type=float, default=0.5, help="预测框与真值匹配IOU阈值")
parser.add_argument("--epoch", type=int, default=14, help="评估的模型epoch")
args = parser.parse_args()

# ====================== 固定配置 ======================
NUM_CLASSES = cfg.NUM_CLASSES + 1  # +1 for background
MODEL_WEIGHT_PATH = os.path.join(cfg.SAVE_WEIGHT_DIR, f"maskrcnn-model-{args.epoch}.pth")
VAL_ROOT = cfg.VOC_ROOT
CONF_THRESH = args.conf
IOU_MATCH_THRESH = args.iou_match
CLASS_LABELS = ["背景", "轻度霉变", "重度霉变"]
OUTPUT_ROOT = "./eval_result_out"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

param_suffix = f"_conf{CONF_THRESH}_iou{IOU_MATCH_THRESH}"
txt_save_path = os.path.join(OUTPUT_ROOT, f"{param_suffix}_metrics.txt")
cm_img_path = os.path.join(OUTPUT_ROOT, f"{param_suffix}_confusion_matrix.jpg")
det_curve_path = os.path.join(OUTPUT_ROOT, f"{param_suffix}_det_curve.jpg")


def calc_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0


def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
img_dir = os.path.join(VAL_ROOT, "JPEGImages", "JPEGImages")  
ann_dir = os.path.join(VAL_ROOT, "Annotations")
txt_path = os.path.join(VAL_ROOT, "ImageSets", "Main", "val.txt")

with open(txt_path, "r", encoding="utf-8") as f:
    val_img_ids = [i.strip() for i in f.readlines()]

# ====================== 加载权重 ======================
print(f"===== 评估 Mask RCNN Epoch {args.epoch} =====")
print(f"CONF_THRESH = {CONF_THRESH}, IOU_MATCH_THRESH = {IOU_MATCH_THRESH}\n")

model = create_model(num_classes=NUM_CLASSES)
assert os.path.exists(MODEL_WEIGHT_PATH), f"权重不存在: {MODEL_WEIGHT_PATH}"
checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=device)
model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
model.to(device)
model.eval()


def get_val_data():
    for img_id in val_img_ids:
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f"{img_id}.png")
        img = Image.open(img_path).convert("RGB")
        ann_path = os.path.join(ann_dir, f"{img_id}.xml")
        tree = ET.parse(ann_path)
        root = tree.getroot()
        boxes = []
        labels = []
        cls_map = {"light_mold": 1, "heavy_mold": 2}
        for obj in root.findall("object"):
            name = obj.find("name").text
            bnd = obj.find("bndbox")
            boxes.append([
                float(bnd.find("xmin").text),
                float(bnd.find("ymin").text),
                float(bnd.find("xmax").text),
                float(bnd.find("ymax").text)
            ])
            labels.append(cls_map.get(name, 1))
        if len(boxes) == 0:
            continue
        target = {"boxes": torch.tensor(boxes), "labels": torch.tensor(labels)}
        yield img, target


gt_label_list = []
pred_label_list = []

with torch.no_grad():
    for img, target_info in get_val_data():
        img_tensor = transform(img).unsqueeze(0).to(device)
        pred_out = model(img_tensor)[0]

        pred_boxes = pred_out["boxes"].cpu().numpy()
        pred_scores = pred_out["scores"].cpu().numpy()
        pred_cls = pred_out["labels"].cpu().numpy()

        valid_mask = pred_scores >= CONF_THRESH
        pred_boxes = pred_boxes[valid_mask]
        pred_cls = pred_cls[valid_mask]
        pred_scores = pred_scores[valid_mask]

        gt_boxes = target_info["boxes"].cpu().numpy()
        gt_cls = target_info["labels"].cpu().numpy()
        matched_gt_index = set()

        for p_box, p_cls in zip(pred_boxes, pred_cls):
            best_iou = 0.0
            best_gt_idx = -1
            for idx, g_box in enumerate(gt_boxes):
                iou = calc_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= IOU_MATCH_THRESH and best_gt_idx not in matched_gt_index:
                matched_gt_index.add(best_gt_idx)
                gt_label_list.append(gt_cls[best_gt_idx])
                pred_label_list.append(p_cls)

        for idx, g_cls in enumerate(gt_cls):
            if idx not in matched_gt_index:
                gt_label_list.append(g_cls)
                pred_label_list.append(0)  # 背景（漏检）

if len(gt_label_list) == 0 or len(pred_label_list) == 0:
    print("无匹配检测样本，退出评估")
    exit()

# 混淆矩阵
conf_matrix = np.zeros((3, 3), dtype=int)
for gt, pred in zip(gt_label_list, pred_label_list):
    conf_matrix[int(gt), int(pred)] += 1

plt.figure(figsize=(9, 7))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title(f"Mask RCNN Epoch {args.epoch}\n(CONF={CONF_THRESH}, IOU={IOU_MATCH_THRESH})")
plt.colorbar()
tick_pos = np.arange(len(CLASS_LABELS))
plt.xticks(tick_pos, CLASS_LABELS, rotation=30)
plt.yticks(tick_pos, CLASS_LABELS)
threshold = conf_matrix.max() / 2
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center",
                 color="white" if conf_matrix[i, j] > threshold else "black")
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.tight_layout()
plt.savefig(cm_img_path)
plt.close()

# 指标计算
lines = [f"==== Mask RCNN Epoch {args.epoch} (CONF={CONF_THRESH}, IOU={IOU_MATCH_THRESH}) ====\n"]
for class_id, class_name in enumerate(CLASS_LABELS):
    if class_id == 0:
        continue
    TP = conf_matrix[class_id, class_id]
    FP = conf_matrix[:, class_id].sum() - TP
    FN = conf_matrix[class_id, :].sum() - TP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    line = f"{class_name} | Precision:{precision:.3f} | Recall:{recall:.3f} | F1:{f1:.3f}"
    lines.append(line)
    print(line)

with open(txt_save_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\n评估完成！")
print(f"混淆矩阵: {cm_img_path}")
print(f"指标文本: {txt_save_path}")
