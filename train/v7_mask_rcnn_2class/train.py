import os
import sys
import datetime
import torch
from torchvision.ops.misc import FrozenBatchNorm2d

# ========== 路径配置：引入同级common_3cls公共库 ==========
cur_path = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.join(cur_path, "..")
sys.path.append(train_root)
cur_path = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.join(cur_path, "..")
# 核心：把train目录加入sys.path，让common_3cls成为顶层包
sys.path.append(train_root)
# 额外把MaskRCNN_utils加入搜索路径（辅助修复导入）
mask_utils_dir = os.path.join(train_root, "common_3cls", "MaskRCNN_utils")
sys.path.append(mask_utils_dir)
# 全局统一配置（v3文件夹内config.py）
import config as cfg
# 公共MaskRCNN网络导入
from common_2cls.MaskRCNN_models.network_files import MaskRCNN
from common_2cls.MaskRCNN_models.backbone import resnet50_fpn_backbone
# 公共MaskRCNN训练工具
from common_2cls.MaskRCNN_utils import train_eval_utils as utils
from common_2cls.MaskRCNN_utils. group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
# 公共数据集、数据增强
from common_2cls.data.my_dataset_voc import VOCInstances
from common_3cls.data import transforms


def create_model(num_classes, load_pretrain_weights=True):
    """构建MaskRCNN，适配项目预训练权重路径"""
    # backbone解冻层数、norm层从cfg读取
    if cfg.TRAINABLE_LAYERS >= 5:
        norm_layer = torch.nn.BatchNorm2d
    else:
        norm_layer = FrozenBatchNorm2d

    backbone = resnet50_fpn_backbone(
        pretrain_path=os.path.join(train_root, "common_3cls", "fasterrcnn_resnet50_fpn_coco.pth"),
        norm_layer=norm_layer,
        trainable_layers=cfg.TRAINABLE_LAYERS
    )

    # 实例分割MaskRCNN主体
    model = MaskRCNN(backbone, num_classes=num_classes)

    if load_pretrain_weights:
        # 加载COCO预训练权重，过滤分类/掩码输出层权重
        maskrcnn_pretrain_path = os.path.join(train_root, "common_3cls", "maskrcnn_resnet50_fpn_coco.pth")
        weights_dict = torch.load(maskrcnn_pretrain_path, map_location="cpu")
        # 移除预测头权重，适配自定义类别数
        filter_keys = []
        for k in list(weights_dict.keys()):
            if "box_predictor" in k or "mask_fcn_logits" in k:
                filter_keys.append(k)
        for k in filter_keys:
            del weights_dict[k]
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        print(f"预训练权重加载完成\nmissing_keys: {missing_keys}\nunexpected_keys: {unexpected_keys}")

    return model


def main(args):
    # 训练设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用训练设备: {device.type}")

    # 日志文件命名
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = os.path.join(cur_path, f"det_results_{time_str}.txt")
    seg_results_file = os.path.join(cur_path, f"seg_results_{time_str}.txt")

    # 数据增强配置（读取cfg）
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5)
        ]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # 数据集路径：统一使用cfg的VOC根目录
    voc_root = cfg.VOC_ROOT
    # 训练集 VOC实例分割数据集
    train_dataset = VOCInstances(voc_root, "2012", "train.txt", data_transform["train"])
    train_sampler = None

    # 长宽比分组采样，减少显存占用
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # DataLoader配置
    batch_size = args.batch_size
    nw = 0  # Mac/Windows设为0，避免多进程报错
    print(f"Dataloader workers = {nw}")
    if train_sampler:
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            pin_memory=True,
            num_workers=nw,
            collate_fn=train_dataset.collate_fn
        )
    else:
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=nw,
            collate_fn=train_dataset.collate_fn
        )

    # 验证集
    val_dataset = VOCInstances(voc_root, "2012", "val.txt", data_transform["val"])
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )

    # 构建模型：NUM_CLASSES=3(不含背景)+1
    model = create_model(num_classes=cfg.NUM_CLASSES + 1, load_pretrain_weights=args.pretrain)
    model.to(device)

    train_loss = []
    learning_rate = []
    val_det_map = []

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_steps,
        gamma=args.lr_gamma
    )

    # 断点续训
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print(f"断点续训，从epoch {args.start_epoch} 开始")

    # 训练循环
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = utils.train_one_epoch(
            model, optimizer, train_data_loader,
            device=device, epoch=epoch, print_freq=50,
            warmup=True, scaler=scaler
        )
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)
        lr_scheduler.step()

        # 验证：检测mAP + 分割mIoU
        det_metrics, seg_metrics = utils.evaluate(model, val_data_loader, device=device)

        # 写入检测指标日志
        with open(det_results_file, "a", encoding="utf-8") as f:
            info = [f"{v:.4f}" for v in det_metrics + [mean_loss.item()]] + [f"{lr:.6f}"]
            f.write(f"epoch:{epoch} {'  '.join(info)}\n")
        # 写入分割指标日志
        with open(seg_results_file, "a", encoding="utf-8") as f:
            info = [f"{v:.4f}" for v in seg_metrics + [mean_loss.item()]] + [f"{lr:.6f}"]
            f.write(f"epoch:{epoch} {'  '.join(info)}\n")

        val_det_map.append(det_metrics[1])

        # 保存权重
        save_dir = cfg.SAVE_WEIGHT_DIR
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        }
        if args.amp:
            save_dict["scaler"] = scaler.state_dict()
        torch.save(save_dict, os.path.join(save_dir, f"maskrcnn-model-{epoch}.pth"))
        print(f"epoch {epoch} 权重已保存至 {save_dir}")

    # 绘制loss-lr曲线
    if train_loss and learning_rate:
        from common_3cls.MaskRCNN_utils.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate, save_path=os.path.join(cur_path, "loss_lr_curve.png"))
    # 绘制mAP曲线
    if val_det_map:
        from common_3cls.MaskRCNN_utils.plot_curve import plot_map
        plot_map(val_det_map, save_path=os.path.join(cur_path, "det_map_curve.png"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="v3 MaskRCNN 玉米霉变实例分割基线训练")

    # 设备
    parser.add_argument('--device', default=cfg.DEVICE, help="训练设备 cuda:0 / cpu")
    # 数据集路径（优先读取cfg，命令行可覆盖）
    parser.add_argument('--data-path', default=cfg.VOC_ROOT, help="VOC数据集根目录")
    # 类别数（固定cfg.NUM_CLASSES=3，命令行仅兼容）
    parser.add_argument('--num-classes', default=cfg.NUM_CLASSES, type=int, help="不含背景的目标类别数")
    # 权重输出目录
    parser.add_argument('--output-dir', default=cfg.SAVE_WEIGHT_DIR, help="权重保存文件夹")
    # 断点续训
    parser.add_argument('--resume', default="", type=str, help="加载历史权重路径")
    parser.add_argument('--start_epoch', default=0, type=int, help="起始epoch")
    # 训练轮次
    parser.add_argument('--epochs', default=cfg.EPOCHS, type=int, help="总训练轮次")
    # 学习率相关
    parser.add_argument('--lr', default=cfg.LEARNING_RATE, type=float, help="初始学习率")
    parser.add_argument('--momentum', default=cfg.MOMENTUM, type=float)
    parser.add_argument('--wd', '--weight-decay', default=cfg.WEIGHT_DECAY, type=float, dest="weight_decay")
    parser.add_argument('--lr-steps', default=[cfg.STEP_SIZE * 2, cfg.STEP_SIZE * 4], nargs="+", type=int)
    parser.add_argument('--lr-gamma', default=cfg.STEP_LR_GAMMA, type=float)
    # 批次与采样
    parser.add_argument('--batch_size', default=cfg.BATCH_SIZE, type=int)
    parser.add_argument('--aspect-ratio-group-factor', default=cfg.ASPECT_RATIO_GROUP, type=int)
    # 预训练与混合精度
    parser.add_argument("--pretrain", type=bool, default=True, help="加载COCO预训练权重")
    parser.add_argument("--amp", default=cfg.USE_AMP, type=bool, help="混合精度训练")

    args = parser.parse_args()
    print("===== 训练超参 =====")
    print(args)

    # 创建权重保存文件夹
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)