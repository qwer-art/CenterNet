import os.path as osp
import sys

import numpy as np

project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)

from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

import tkinter as tk

def test_transform():
    image_path = '/home/zyt/Data/VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg'
    # 1.pil_image
    image = Image.open(image_path)
    width, height = image.size
    print(f"pil_image: ({height}x{width}),min: {np.min(image)},max: {np.max(image)}")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    # 2.tf_image
    transformed_image = transform(image)
    print(f"tf_image,{transformed_image.shape},min: {transformed_image.min()},max: {transformed_image.max()}")
    # 3.tf_image_to_pil_image
    transformed_image_pil = transforms.ToPILImage()(transformed_image)
    width, height = transformed_image_pil.size
    print(f"tf_to_pil: ({height}x{width}),min:{np.min(transformed_image_pil)},max: {np.max(transformed_image_pil)}")
    # show
    plt.imshow(transformed_image_pil)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

if __name__ == '__main__':
    test_transform()