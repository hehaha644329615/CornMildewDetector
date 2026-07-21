"""
在真实验证集图片上绘制 GT（绿色） vs 预测框（红色）
"""
import os, random, torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
import matplotlib.pyplot as plt

val_img_dir = "../common_3cls/data/YOLO_data/images/val"
val_label_dir = "../common_3cls/data/YOLO_data/labels/val"
model = YOLO('path/to/best.pt')
img_files = random.sample(os.listdir(val_img_dir), 6)

fig, axes = plt.subplots(2, 3, figsize=(15,10))
for ax, f in zip(axes.flat, img_files):
    img = Image.open(os.path.join(val_img_dir, f))
    results = model(img)
    
    draw = ImageDraw.Draw(img)
    # 画 GT 绿框
    label_file = os.path.join(val_label_dir, f.replace('.jpg','.txt'))
    if os.path.exists(label_file):
        for line in open(label_file):
            c, cx, cy, w, h = map(float, line.split())
            x1 = (cx - w/2) * img.width
            y1 = (cy - h/2) * img.height
            x2 = (cx + w/2) * img.width
            y2 = (cy + h/2) * img.height
            draw.rectangle([x1,y1,x2,y2], outline='green', width=2)
    # 画预测红框
    for box in results[0].boxes.xyxy:
        draw.rectangle(box.tolist(), outline='red', width=2)
    
    ax.imshow(img)
    ax.axis('off')

plt.savefig('docs/images/gt_vs_pred.png')