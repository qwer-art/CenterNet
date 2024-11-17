import os.path as osp
import sys
project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def image_path_to_pil_image(image_path):
    image = Image.open(image_path)
    return image

def pil_image_to_tf_image(pil_image,transform):
    tf_image = transform(pil_image)
    return tf_image

def norm_image_to_denorm_image(norm_image,mean,std):
    mean = torch.tensor(mean).view(-1, 1, 1)  # (C, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)  # (C, 1, 1)
    denorm_image = norm_image * std + mean
    return denorm_image

def tensor_image_to_visual_image(tensor_image):
    # type to int
    visual_image = (tensor_image * 255).clamp(0, 255).byte()
    # 转置形状从 (C, H, W) 到 (H, W, C)
    recovered_image = np.transpose(visual_image, (1, 2, 0))  # 变为 (224, 224, 3)
    return recovered_image

def test_image_transform():
    # 1.image_path -> image
    image_path = '/home/zyt/Data/VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg'
    pil_image = image_path_to_pil_image(image_path)
    # 2.pil_image to tf_image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Resize(512),
        transforms.CenterCrop(512)
    ])
    tf_image = pil_image_to_tf_image(pil_image,transform)
    # 3.norm_image to denorm image
    denorm_image = norm_image_to_denorm_image(tf_image,mean,std)
    # 4.denorm_image to visual image
    visual_image = tensor_image_to_visual_image(denorm_image)

    plt.imshow(visual_image)
    plt.show()

if __name__ == '__main__':
    test_image_transform()
