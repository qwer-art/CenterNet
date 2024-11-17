import os.path as osp
import sys
project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)

from utils.util import *
from pycocotools.coco import COCO
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CocoDataset(Dataset):
    def __init__(self, ann_file, img_dir, transform=None):
        """
        初始化 COCO 数据集类
        :param ann_file: COCO 注释文件的路径 (例如: 'annotations/instances_train2017.json')
        :param img_dir: 图像所在的文件夹路径 (例如: 'train2017/')
        :param transform: 可选的图像转换（如数据增强等）
        """
        self.coco = COCO(ann_file)  # 加载 COCO 注释文件
        self.img_dir = img_dir      # 图像文件夹
        self.transform = transform  # 图像转换
        self.img_ids = list(self.coco.imgs.keys())  # 获取所有图像的 ID 列表

    def __len__(self):
        """返回数据集的大小"""
        return len(self.img_ids)

    def __getitem__(self, idx):
        """返回一个样本（图像和目标框）"""
        # 获取图像ID
        img_id = [self.img_ids[idx]]

        # 加载图像
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # 获取图像的标注信息（目标框、类别等）
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # 获取目标框和类别标签
        boxes = []
        labels = []
        for ann in anns:
            # COCO中的目标框格式是[x, y, width, height]
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])

        # 将目标框转换为Tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # 如果有定义transform（数据增强或预处理），则应用
        if self.transform:
            img,boxes = self.transform(img,boxes)

        return img, boxes, labels

def test_dataset():
    # 1.dataset input
    ann_file = coco_anno_file
    img_dir = dataset_image_path

    # 创建数据集对象
    coco_dataset = CocoDataset(ann_file=ann_file, img_dir=img_dir, transform=transform)
    print(f"dataset: {len(coco_dataset)}")

    # 创建DataLoader
    data_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=4, shuffle=True)

    idx = 0
    img, boxes, labels = coco_dataset[idx]
    print(f"img: {img.shape}")
    print(f"boxes: {boxes.shape}")
    print(f"labels: {labels.shape}")

if __name__ == '__main__':
    test_dataset()
