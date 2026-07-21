import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.join(cur_path, "..")
sys.path.append(train_root)

import time
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import config as cfg

from common_2cls.MaskRCNN_models.network_files import MaskRCNN
from common_2cls.MaskRCNN_models.backbone import resnet50_fpn_backbone
from common_2cls.MaskRCNN_utils.draw_box_utils import draw_objs


def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"推理设备：{device}")

    num_classes = cfg.NUM_CLASSES + 1  # +1 for background
    model = create_model(num_classes=num_classes)

    weights_path = os.path.join(cfg.SAVE_WEIGHT_DIR, "maskrcnn-model-14.pth")
    assert os.path.exists(weights_path), f"权重不存在: {weights_path}"
    weights_dict = torch.load(weights_path, map_location="cpu")
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)
    print("权重加载完成")

    json_path = cfg.JSON_CLASS_PATH
    assert os.path.exists(json_path), f"类别文件不存在: {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        class_dict = json.load(f)
    category_index = {str(v): str(k) for k, v in class_dict.items()}

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
        predict_masks = predictions["masks"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("未检测到目标！")
            return

    # 绘制检测框
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

    # 叠加分割掩码（Mask RCNN 专属）
    if len(predict_masks) > 0:
        import numpy as np
        mask_overlay = np.array(plot_img).copy()
        for i, mask in enumerate(predict_masks):
            if predict_scores[i] >= cfg.BOX_THRESHOLD:
                mask_binary = mask[0] > 0.5
                color = [(255, 255, 0), (255, 0, 0)][predict_classes[i] - 1]
                mask_overlay[mask_binary] = mask_overlay[mask_binary] * 0.5 + np.array(color) * 0.5
        plot_img = Image.fromarray(mask_overlay.astype(np.uint8))

    plt.figure(figsize=(12, 8))
    plt.imshow(plot_img)
    plt.axis("off")
    plt.show()
    save_result = os.path.join(cur_path, "predict_result.jpg")
    plot_img.save(save_result)
    print(f"预测结果保存至: {save_result}")


if __name__ == '__main__':
    main()