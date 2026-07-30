# 🌽 玉米霉变检测系统

> **基于深度学习的玉米籽粒霉变实时检测，6 版本迭代，从 Faster R-CNN 到 YOLOv8，最终落地 iOS App。**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![iOS](https://img.shields.io/badge/Platform-iOS-lightgrey.svg)]()

---

## 📖 项目简介

本项目源于饲料厂收购玉米时的真实需求：快速、准确地判断玉米籽粒霉变率，用于定价与收拒决策。传统人工目测效率低、主观性强、无可追溯记录。

本系统通过 **深度学习检测 + 硬件标准化采集 + 端侧部署**，实现从拍照到输出霉变率的自动化流程。

**完整版本故事**：[📝 项目版本演进全记录](docs/PROJECT_STORY.md)

---

## 🎯 核心功能

- **实时检测**：YOLOv8n 在 iPhone 14 Pro 上约 22 FPS
- **二分类识别**：轻度霉变 / 重度霉变（最终方案已去掉健康类别）
- **霉变率计算**：按业务规则自动统计并换算霉变率
- **标准化采集**：30×40cm 木盒 + 手机支架，统一距离与角度
- **离线运行**：INT8 量化后约 3.8MB，适合收购现场弱网环境

---

## 🏗️ 技术栈

| 环节 | 技术 |
|:---|:---|
| 模型训练 | PyTorch, Ultralytics YOLOv8, Faster R-CNN |
| 模型部署 | ONNX, CoreML, INT8 量化 |
| 移动端 | SwiftUI, AVFoundation, Vision |
| 数据处理 | Label Studio, OpenCV, Albumentations |
| 硬件方案 | 30×40cm 木盒 + 定制手机支架 |

---

## 📊 最终效果（v6）

| 指标 | 数值 |
|:---|:---|
| mAP50 | **0.934** |
| 推理速度 | **22 FPS**（iPhone 14 Pro） |
| 模型大小 | **3.8 MB**（INT8 量化后） |
| 漏检率 | < 5% |

---

## 📁 仓库结构

```text
CornMildewDetector/
├── docs/                  # 项目故事、部署指南、技术细节
├── train/                 # v1–v6 训练与导出代码
│   ├── v1_faster_rcnn_baseline/
│   ├── v2_faster_rcnn_focal/
│   ├── v3_yolov8_baseline/
│   ├── v4_yolov8_optimized/
│   ├── v5_yolov8_fixed_camera/
│   └── v6_final/          # 最终上线版
├── app/                   # iOS 核心源码
├── hardware/              # 木盒设计图与组装说明
├── LICENSE
└── README.md
```

---

## 🚀 快速开始

### 1. 环境

建议 Python 3.10+：

```bash
pip install ultralytics torch torchvision opencv-python coremltools
```

（若后续补充了 `train/requirements.txt`，优先使用该文件。）

### 2. 数据挂载

训练数据放在仓库外，通过软链接挂载。以二分类 YOLO 数据为例：

```bash
cd train/common_2cls/data
ln -s ../../../../Two_Class_Data/YOLO_data ./YOLO_data
```

详见各 `data/README.md`。

### 3. 训练（最终版）

```bash
cd train/v6_final
python train.py
```

### 4. 导出与部署

```bash
python export_onnx.py
python quantize.py
```

将生成的 CoreML 模型放入 iOS 工程，真机运行。更多说明见：

- [部署指南](docs/DEPLOYMENT_GUIDE.md)
- [技术细节](docs/TECHNICAL_DETAILS.md)
- [App 说明](app/README.md)

---

## 🔁 版本演进一览

| 版本 | 模型 | 类别 | mAP50 | 核心改进 |
|:---|:---|:---:|:---:|:---|
| v1 | Faster R-CNN | 3 | ~0.242 | 基线，发现健康样本噪声 |
| v2 | Faster R-CNN + Focal | 3 | ~0.263 | Focal Loss，治标不治本 |
| v3 | YOLOv8n | 2 | 0.277 | 轻量化，去掉健康类别 |
| v4 | YOLOv8n | 2 | 0.293 | 修复数据泄露 + 背景负样本 |
| v5 | YOLOv8n | 2 | 0.920 | **硬件标准化采集** |
| **v6** | **YOLOv8n + INT8** | **2** | **0.934** | 增强数据 + 量化上线 |

---

## 📌 说明

- 原项目为企业内部项目，本仓库为可公开的复现与展示版本
- 完整业务数据与部分权重未开源；训练需自行准备/挂载数据
- App 目录提供核心 Swift 源码，完整 Xcode 工程需按说明自行配置

---

## 📄 License

[MIT](LICENSE)