### v2 为 v1 原版 Faster R-CNN（交叉熵 CE 损失）的改进版，核心改动：将 ROI 检测头多分类损失、RPN 前景背景二分类损失替换为 Focal Loss，解决玉米霉变检测任务中正负样本不均衡、难例样本识别差的问题。
    v1：标准 Faster R-CNN + CrossEntropy / BCE 损失（基线版本）
    v2：Faster R-CNN + FocalLoss（ROI 多分类）+ BinaryFocalLoss（RPN 二分类）（优化版本）
两套版本代码物理隔离，公共原版代码无修改，实验基线互不污染。

v2_faster_rcnn_focal/
├── config.py                # 全局路径、超参、Focal超参统一配置
├── train.py                 # 训练入口（加载focal版RPN/ROI头）
├── predict.py               # 单张图片推理可视化
├── eval.py                  # 验证集批量评估：混淆矩阵 + Precision/Recall/F1指标
├── requirements.txt         # 环境依赖
├── frcnn_env.yml            # Conda环境导出配置
├── save_weights/            # 训练权重自动保存目录
├── eval_result_out/         # eval.py 输出：混淆矩阵图、指标txt
└── test.jpg                 # 测试样例图

### 核心修改点（公共 common_3cls）
    - 仅新增_focal后缀副本，原版文件完全不动：
    - loss_focal.py：实现多分类 FocalLoss、二分类 BinaryFocalLoss
    - roi_head_focal.py：替换 ROI 分类交叉熵为 FocalLoss
    - rpn_function_focal.py：替换 RPN 前景背景 BCE 为 BinaryFocalLoss
边框回归仍使用 SmoothL1，无改动