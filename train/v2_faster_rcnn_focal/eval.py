import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from predict import create_model
import xml.etree.ElementTree as ET
import argparse
import config as cfg

# 解决matplotlib中文乱码
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 命令行动态传参（先解析args） ======================
parser = argparse.ArgumentParser(description="仅评估Epoch14最优权重，命令行传入置信、IOU匹配阈值")
parser.add_argument("--conf", type=float, default=0.25, help="检测框置信过滤阈值 CONF_THRESH")
parser.add_argument("--iou_match", type=float, default=0.4, help="预测框与真值匹配IOU阈值 IOU_MATCH_THRESH")
args = parser.parse_args()

# 读取参数
NUM_CLASSES = cfg.NUM_CLASSES
CONF_THRESH = args.conf
IOU_MATCH_THRESH = args.iou_match

# ====================== 全局固定配置 ======================
# 读取Epoch14权重完整路径
MODEL_WEIGHT_PATH = "./save_weights/resNetFpn-model-14.pth"
VAL_ROOT = "./dataset/VOCdevkit/VOC2012"
CLASS_LABELS = ["背景", "健康", "轻度霉变", "重度霉变"]
# 固定单层输出总文件夹，不再拼接阈值后缀
OUTPUT_ROOT = "./eval_result_out"
# 自动创建顶层文件夹，不存在则新建
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 文件名携带阈值标识，区分多组消融实验
param_suffix = f"_conf{CONF_THRESH}_iou{IOU_MATCH_THRESH}"
txt_save_path = os.path.join(OUTPUT_ROOT, f"{param_suffix}_metrics.txt")
cm_img_path = os.path.join(OUTPUT_ROOT, f"{param_suffix}_confusion_matrix.jpg")

# =================================================================
# IOU计算工具函数
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

# 全局基础路径初始化
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
img_dir = os.path.join(VAL_ROOT, "JPEGImages")
ann_dir = os.path.join(VAL_ROOT, "Annotations")
txt_path = os.path.join(VAL_ROOT, "ImageSets", "Main", "val.txt")

# 一次性读取全部验证集图片ID
with open(txt_path, "r", encoding="utf-8") as f:
    val_img_ids = [i.strip() for i in f.readlines()]

# ====================== 加载Epoch14固定权重 ======================
print(f"===== 开始评估 Epoch 14 权重文件：{MODEL_WEIGHT_PATH} =====")
print(f"当前实验参数：CONF_THRESH = {CONF_THRESH}, IOU_MATCH_THRESH = {IOU_MATCH_THRESH}\n")

model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

# 验证集生成器，每次全新读取标注，避免迭代器耗尽bug
def get_val_data():
    for img_id in val_img_ids:
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        tree = ET.parse(os.path.join(ann_dir, f"{img_id}.xml"))
        root = tree.getroot()
        boxes = []
        labels = []
        cls_map = {"healthy": 1, "light_mold": 2, "heavy_mold": 3}
        for obj in root.findall("object"):
            name = obj.find("name").text
            bnd = obj.find("bndbox")
            x1 = float(bnd.find("xmin").text)
            y1 = float(bnd.find("ymin").text)
            x2 = float(bnd.find("xmax").text)
            y2 = float(bnd.find("ymax").text)
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_map[name])
        target = {
            "boxes": torch.tensor(boxes),
            "labels": torch.tensor(labels)
        }
        yield img, target

# 存储全部真值、预测类别，用于统计混淆矩阵
gt_label_list = []
pred_label_list = []

# 推理关闭梯度，节省显存
with torch.no_grad():
    for img, target_info in get_val_data():
        img_tensor = transform(img).unsqueeze(0).to(device)
        pred_out = model(img_tensor)[0]

        pred_boxes = pred_out["boxes"].cpu().numpy()
        pred_scores = pred_out["scores"].cpu().numpy()
        pred_cls = pred_out["labels"].cpu().numpy()

        # 按置信阈值过滤低置信预测框
        valid_mask = pred_scores >= CONF_THRESH
        pred_boxes = pred_boxes[valid_mask]
        pred_cls = pred_cls[valid_mask]
        pred_scores = pred_scores[valid_mask]

        gt_boxes = target_info["boxes"].cpu().numpy()
        gt_cls = target_info["labels"].cpu().numpy()
        matched_gt_index = set()

        # 预测框与真值IOU匹配逻辑
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
        # 未匹配真值标记为漏检（背景0）
        for idx, g_cls in enumerate(gt_cls):
            if idx not in matched_gt_index:
                gt_label_list.append(g_cls)
                pred_label_list.append(0)

# 兜底判断：无有效样本直接终止
if len(gt_label_list) == 0 or len(pred_label_list) == 0:
    print("无匹配检测样本，不生成图表与指标文件")
    exit()

# 手动构建4×4混淆矩阵
conf_matrix = np.zeros((4, 4), dtype=int)
for gt, pred in zip(gt_label_list, pred_label_list):
    conf_matrix[int(gt), int(pred)] += 1

# 绘制混淆矩阵，标题自动打印当前阈值
plt.figure(figsize=(9,7))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title(f"Epoch 14 Confusion Matrix\n(CONF_THRESH = {CONF_THRESH}, IOU_MATCH_THRESH = {IOU_MATCH_THRESH}) ,v2")
plt.colorbar()
tick_pos = np.arange(len(CLASS_LABELS))
plt.xticks(tick_pos, CLASS_LABELS, rotation=30)
plt.yticks(tick_pos, CLASS_LABELS)
threshold = conf_matrix.max() / 2
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, str(conf_matrix[i,j]), ha="center", va="center",
                 color="white" if conf_matrix[i,j] > threshold else "black")
plt.xlabel("预测类别 Pred Label")
plt.ylabel("真实类别 True Label")
plt.tight_layout()
plt.savefig(cm_img_path)
plt.close()

# 计算每类Precision / Recall / F1，写入txt
lines = [
    f"==== Epoch 14 评价指标 (CONF_THRESH = {CONF_THRESH}, IOU_MATCH_THRESH = {IOU_MATCH_THRESH}) ,v2====\n"
]
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

# 完成打印
print(f"\n评估全部完成！")
print(f"混淆矩阵图片：{cm_img_path}")
print(f"量化指标文本：{txt_save_path}")