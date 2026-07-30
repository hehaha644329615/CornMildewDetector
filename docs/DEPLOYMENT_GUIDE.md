# iOS 端到端部署指南

本文档涵盖从训练好的YOLOv8模型导出、iOS工程配置，到真机运行与硬件使用的完整流程。


## 1. 模型导出与转换 (MacOS)

在完成模型训练后，需要将其转换为iOS可用的CoreML格式。

### 1.1 PyTorch → ONNX

在训练环境（如 `train/v6_final/`）下执行：

```bash
# 使用Ultralytics命令
yolo export model=path/to/best.pt format=onnx imgsz=640 simplify=True
```

或运行项目提供的导出脚本：

```bash
python export_onnx.py
```

### 1.2 ONNX → CoreML (INT8量化)

使用 `coremltools` 对ONNX模型进行量化，以缩小体积并提升推理速度。执行量化脚本：

```bash
python quantize.py
```

最终得到 **`best.mlpackage`** 或 **`best.mlmodel`**，体积约 **3.8 MB**。


## 2. iOS工程配置 (Xcode)

1. **打开项目**：在Xcode中打开 `ios_app/CornMildewDetector.xcodeproj`

2. **添加模型**：将上一步量化好的 `.mlpackage` 文件拖入Xcode工程，并在弹窗中**勾选目标的Target**，确保模型会被打包进App

3. **配置权限**：在 `Info.plist` 文件中添加摄像头权限描述：
   - Key: `Privacy - Camera Usage Description`
   - Value: `用于玉米霉变实时拍照检测`

4. **选择开发者账号**：在项目设置 `Signing & Capabilities` 中，选择你的Apple开发者账号（Team）。如需真机运行，必须使用有效的开发者账号


## 3. 核心代码对应关系

为了方便排查问题，理解代码模块的职责：

| 功能模块 | 对应文件 | 主要职责 |
| :--- | :--- | :--- |
| **相机采集** | `CameraViewController` | 管理AVCaptureSession，获取视频流，输出`CVPixelBuffer` |
| **模型推理** | `DetectorService.swift` | 加载CoreML模型，执行`VNCoreMLRequest`，获取检测结果 |
| **坐标换算** | `DetectorService.swift` | 将模型输出的归一化坐标修正为显示坐标（处理黑边、旋转） |
| **霉变统计** | `MildewCalculator.swift` | 统计轻/重度霉变框数量，根据公式计算霉变率 |
| **UI绘制** | `DetectionOverlayView` (SwiftUI) | 在相机预览层上绘制半透明检测框和标签 |


## 4. 硬件使用规范（关键）

**必须严格遵守**，否则检测精度将显著下降：

1. **标准容器**：使用项目提供的 `300×400×40 mm` 木盒

2. **固定设备**：将手机固定在木盒上方的支架上，确保镜头**垂直向下**，拍摄距离和角度固定

3. **均匀铺样**：每次检测约**一斤**玉米，倒入木盒后均匀摊开，确保**单层、无重叠**

4. **触发拍照**：使用蓝牙遥控器或App内按钮拍照，避免触碰手机造成晃动


## 5. 霉变率计算逻辑

App内部计算逻辑如下，分母与拍照条件共同保证了结果的稳定性：

- **公式**：霉变率 = (x₁/2 + x₂) / 5μ

- **参数**：
  - x₁：轻度霉变检测框数量
  - x₂：重度霉变检测框数量
  - μ：100g 玉米平均粒数（上线版取 **120**）

- **依据**：一斤玉米 ≈ 5 × μ 粒


## 6. 真机运行与常见问题

### 前置条件

- Mac电脑 + Xcode 15及以上
- **真机运行**（模拟器无法使用相机），需Apple开发者账号
- 已导出符合要求的 `best.mlpackage` 模型

### 常见问题排查

| 问题 | 可能原因 | 处理建议 |
| :--- | :--- | :--- |
| **摄像头黑屏/无画面** | 1. 权限未配置或未生效<br>2. Xcode权限缓存 | 1. 检查 `Info.plist` 权限描述<br>2. 重启Xcode，**卸载重装**App |
| **检测框错位/偏移** | 输入尺寸与显示尺寸不匹配，或黑边（Letterbox）坐标换算公式错误 | 检查 `DetectorService` 中的 `convertToImageCoordinates` 和 `centerCrop` 设置 |
| **推理速度卡顿** | 1. 未开启GPU加速<br>2. 模型未做INT8量化 | 1. 确认使用 `VNCoreMLRequest` 的默认GPU调用<br>2. **务必**使用INT8量化后的模型 |
| **App闪退** | 1. 模型文件损坏或不兼容<br>2. 输入图片分辨率与模型尺寸不匹配 | 1. 重新导出并导入模型<br>2. 确认输入尺寸为 `640×640` |
| **模型加载失败** | 模型文件未正确加入Target | 检查 `Build Phases` → `Copy Bundle Resources` 中是否包含 `.mlpackage` |
| **检测框乱跳** | 手机未固定，拍摄时晃动 | **必须使用硬件支架**进行拍摄 |


## 7. 部署检查清单

在正式使用前，建议逐项确认：

- [ ] 已导出并量化 `CoreML` 模型（`best.mlpackage`）
- [ ] 模型已正确加入Xcode工程并勾选Target
- [ ] `Info.plist` 已添加相机权限描述
- [ ] 硬件支架高度与角度已固定，手机夹紧
- [ ] 已使用真实玉米样本做至少5组对比测试，验证精度
- [ ] （如需分发）已使用开发者账号Archive打包，导出 `.ipa` 文件