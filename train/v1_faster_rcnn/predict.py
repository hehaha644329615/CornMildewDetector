import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.join(cur_path, "..")
sys.path.append(train_root)

# 剩余基础库导入
import time
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import config as cfg

# 公共模块导入
from common_3cls.FasterRCNN_models.network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from common_3cls.FasterRCNN_models.backbone import resnet50_fpn_backbone, MobileNetV2
from common_3cls.FasterRCNN_utils.draw_box_utils import draw_objs


def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"推理设备：{device}")

    num_classes = cfg.NUM_CLASSES + 1  # 加1是因为类别编号从1开始，0是背景
    model = create_model(num_classes=num_classes)

    # 权重路径
    weights_path = os.path.join(cfg.SAVE_WEIGHT_DIR, "model.pth")
    assert os.path.exists(weights_path), f"权重不存在: {weights_path}"
    weights_dict = torch.load(weights_path, map_location="cpu")
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)
    print("权重加载完成")

    # 类别json路径
    json_path = cfg.JSON_CLASS_PATH
    assert os.path.exists(json_path), f"类别文件不存在: {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        class_dict = json.load(f)
    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # 测试图片
    test_img_path = os.path.join(cur_path, "test.jpg")
    assert os.path.exists(test_img_path), f"测试图片不存在: {test_img_path}"
    original_img = Image.open(test_img_path).convert("RGB")

    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        img_h, img_w = img.shape[-2:]
        warmup_img = torch.zeros((1, 3, img_h, img_w), device=device)
        model(warmup_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print(f"推理耗时: {t_end - t_start:.4f} s")

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("未检测到目标！")
            return

    # 绘制框，Mac字体兼容
    try:
        plot_img = draw_objs(
            original_img, predict_boxes, predict_classes, predict_scores,
            category_index=category_index, box_thresh=cfg.BOX_THRESHOLD,
            line_thickness=3, font='arial.ttf', font_size=20
        )
    except:
        plot_img = draw_objs(
            original_img, predict_boxes, predict_classes, predict_scores,
            category_index=category_index, box_thresh=cfg.BOX_THRESHOLD,
            line_thickness=3, font='/System/Library/Fonts/PingFang.ttc', font_size=20
        )

    plt.figure(figsize=(12, 8))
    plt.imshow(plot_img)
    plt.axis("off")
    plt.show()
    save_result = os.path.join(cur_path, "predict_result.jpg")
    plot_img.save(save_result)
    print(f"预测结果保存至: {save_result}")


if __name__ == '__main__':
    main()