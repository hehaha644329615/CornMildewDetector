# 二分类数据集挂载说明

本目录通过软链接挂载仓库外部的二分类数据集，避免把大数据直接提交进 Git。

当前训练代码（v3–v6）主要使用：

YOLO_data/  →  ../../../../Two_Class_Data/YOLO_data

新设备重建软链接（在本 data 目录执行）

cd train/common_2cls/data

# 清理旧链接（可选）
rm -f YOLO_data VOC2012 YOLO_txt COCO

# 必做：YOLOv8 训练数据
ln -s ../../../../Two_Class_Data/YOLO_data ./YOLO_data