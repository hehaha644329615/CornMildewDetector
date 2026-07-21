import os
import random

def main():
    random.seed(0)
    # 1. 获取当前脚本my_split_data.py所在目录：common_3cls/data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. 往上两级：script_dir = .../train/common_3cls/data
    #    上一级 common_3cls，再上一级 train 文件夹
    train_root = os.path.dirname(os.path.dirname(script_dir))
    # 3. 软链接地址：train/common_3cls/data/VOC2012
    voc_link_root = os.path.join(script_dir, "VOC2012")
    ann_path = os.path.join(voc_link_root, "Annotations")

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
    assert total > 0, "Annotations文件夹下没有xml标注文件！"

    # 随机选验证集下标
    val_idx = set(random.sample(range(total), k=int(total * val_ratio)))
    train_list, val_list = [], []
    for i, name in enumerate(file_names):
        if i in val_idx:
            val_list.append(name)
        else:
            train_list.append(name)

    # txt保存路径
    save_folder = os.path.join(voc_link_root, "ImageSets/Main")
    os.makedirs(save_folder, exist_ok=True)
    train_txt = os.path.join(save_folder, "train.txt")
    val_txt = os.path.join(save_folder, "val.txt")

    # 写入文件
    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(train_list))
    with open(val_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(val_list))

    print(f"划分完成：\n训练集 {len(train_list)} 张\n验证集 {len(val_list)} 张")
    print(f"train.txt、val.txt 生成路径：{save_folder}")

if __name__ == "__main__":
    main()