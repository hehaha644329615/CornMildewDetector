# 三分类数据集挂载说明
本目录3个文件夹均为软链接，指向项目外层 Three_Class_Data 下对应格式数据集：
1. VOC2012  —— Faster RCNN、Mask RCNN 训练读取
2. YOLO_txt —— YOLOv8 系列训练读取
3. COCO     —— COCO json评测、分割指标计算

## 新设备重建链接命令（进入本data目录执行）
ln -s ../../../../Three_Class_Data/VOC2012 ./VOC2012
ln -s ../../../../Three_Class_Data/YOLO_txt ./YOLO_txt
ln -s ../../../../Three_Class_Data/COCO ./COCO
<!-- 
../ → common_3cls
../../ → train
../../../ → CornMildewDetector（仓库根目录）
../../../../ → 顶层「玉米检测项目」，和 Three_Class_Data 同级 
-->