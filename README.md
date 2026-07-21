# 🌽 玉米霉变检测系统

> **基于深度学习的玉米籽粒霉变实时检测，6 版本迭代，从 Faster RCNN 到 YOLOv8，最终落地 iOS App。**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![iOS](https://img.shields.io/badge/Platform-iOS-lightgrey.svg)]()

---

## 📖 项目简介

本项目源于饲料厂收购玉米时的真实需求：快速、准确地检测玉米籽粒的霉变率。传统人工目测效率低、主观性强，本系统通过深度学习 + 标准化采集方案，实现了从拍照到出结果的全自动化。

**完整版本故事**：[📝 项目版本演进全记录](docs/PROJECT_STORY.md)

---

## 🎯 核心功能

- **实时检测**：YOLOv8 在 iPhone 14 Pro 上达到 22 FPS
- **精准分类**：健康 / 轻度霉变 / 重度霉变 三类识别
- **霉变率计算**：自动统计霉变颗粒数，计算霉变率
- **标准化采集**：30×40cm 木盒 + 手机支架，统一拍照标准

---

## 🏗️ 技术栈

| 环节 | 技术 |
|:---|:---|
| 模型训练 | PyTorch, YOLOv8, Faster RCNN, Mask RCNN |
| 模型部署 | CoreML, ONNX, 量化剪枝 |
| 移动端 | SwiftUI, AVFoundation, Vision |
| 硬件方案 | 30×40cm 木盒 + 定制手机支架 |

---

## 📊 最终效果

| 指标 | 数值 |
|:---|:---|
| mAP50 | 0.93 |
| 推理速度 | 22 FPS（iPhone 14 Pro） |
| 模型大小 | 3.8 MB（INT8量化后） |
| 漏检率 | < 5% |

---

## 🚀 快速开始

### 环境配置

```bash
pip install -r train/requirements.txt
