# 三分类数据集挂载说明
本目录3个文件夹均为软链接，指向项目外层 Three_Class_Data 下对应格式数据集：

1. VOC2012  —— Faster RCNN 训练读取
2. YOLO_txt —— YOLOv8 系列训练读取

## 新设备重建链接命令（进入本data目录执行）
ln -s ../../../../Three_Class_Data/VOC2012 ./VOC2012
ln -s ../../../../Three_Class_Data/YOLO_txt ./YOLO_txt
ln -s ../../../../Three_Class_Data/COCO ./COCO
