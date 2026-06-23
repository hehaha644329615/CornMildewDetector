# 二分类数据集挂载说明
当前目录三个文件夹均为软链接，指向项目外层 Two_Class_Data 数据集：
1. VOC2012  —— Mask RCNN 训练专用
2. COCO     —— 计算COCO mAP、分割指标评测使用

## 新设备重建软链接命令（在当前data文件夹执行）
ln -s ../../../../Two_Class_Data/VOC2012 ./VOC2012
ln -s ../../../../Two_Class_Data/YOLO_txt ./YOLO_txt
ln -s ../../../../Two_Class_Data/COCO ./COCO
