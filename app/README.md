# iOS App 说明

## 功能概述

- 实时视频流检测（约 22 FPS）
- 轻度霉变（橙色框）/ 重度霉变（红色框）可视化
- 自动计算并显示霉变率
- 支持拍照留底（可扩展 GPS 与时间记录）

## 技术栈

- SwiftUI
- AVFoundation（相机）
- Vision / CoreML（模型推理）

## 目录结构
app/
├── README.md
└── CornMildewDetector/
├── CornMildewDetectorApp.swift   # App 入口
├── Services/
│   └── DetectorService.swift     # 模型加载与推理
└── Views/
├── ContentView.swift
└── CameraView.swift


## 运行要求

- 真机（必须）
- iOS 15.0+
- Xcode 14+
- 已配置相机权限

## 如何替换模型

1. 将量化后的 CoreML 模型（`.mlmodel` / `.mlpackage`）拖入工程
2. 在 `DetectorService.swift` 中确认模型名称与加载代码一致
3. 重新编译运行

## 注意事项

- 本仓库目前主要提供核心源码，完整 Xcode 工程文件可能需要自行创建或补充
- 检测效果严重依赖硬件标准化，请配合 `hardware/` 目录下的支架使用
- 最终业务逻辑以 `docs/PROJECT_STORY.md` 为准（二分类 + 固定分母计算霉变率）