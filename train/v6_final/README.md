```markdown
# v6 最终上线版

YOLOv8n + 数据增强 + 空背景负样本 + INT8 量化，最终部署到 iOS App。

## 核心指标

| 项目 | 数值 |
|------|------|
| 模型 | YOLOv8n |
| 类别 | 2（light_mold / heavy_mold） |
| mAP50 | **0.934** |
| 漏检率 | < 5% |
| 模型大小 | **3.8 MB**（INT8） |
| 推理速度 | **22 FPS**（iPhone 14 Pro） |

## 相对 v5 的改进

- 扩大固定支架下的训练数据，并做数据增强
- 加入空背景负样本，降低背景虚警
- INT8 量化 + 导出 CoreML，完成端侧部署

## 快速使用

```bash
# 训练
python train.py

# 导出 ONNX / CoreML（见本目录脚本）
python export_onnx.py
python quantize.py
```

模型权重建议路径：`v6_final/production/weights/best.pt`

## 相关文档

- 完整迭代故事：[`docs/PROJECT_STORY.md`](../../docs/PROJECT_STORY.md)
- 部署说明：[`docs/DEPLOYMENT_GUIDE.md`](../../docs/DEPLOYMENT_GUIDE.md)
```