import os
import random

def main():
    random.seed(0)
    ann_path = "./dataset/VOCdevkit/VOC2012/Annotations"
    assert os.path.exists(ann_path), f"标注文件夹不存在：{ann_path}"

    # 8:2划分
    val_ratio = 0.2

    # 只筛选xml文件，提取纯文件名
    file_names = []
    for f in os.listdir(ann_path):
        if f.endswith(".xml"):
            file_names.append(f[:-4])
    file_names = sorted(file_names)
    total = len(file_names)

    # 随机选验证集下标
    val_idx = set(random.sample(range(total), k=int(total * val_ratio)))
    train_list, val_list = [], []
    for i, name in enumerate(file_names):
        if i in val_idx:
            val_list.append(name)
        else:
            train_list.append(name)

    # 输出到ImageSets/Main
    save_folder = "./dataset/VOCdevkit/VOC2012/ImageSets/Main"
    os.makedirs(save_folder, exist_ok=True)
    train_txt = os.path.join(save_folder, "train.txt")
    val_txt = os.path.join(save_folder, "val.txt")

    # 写入文件
    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(train_list))
    with open(val_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(val_list))

    print(f"划分完成：\n训练集 {len(train_list)} 张\n验证集 {len(val_list)} 张")

if __name__ == "__main__":
    main()