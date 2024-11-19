import os.path as osp
import sys
project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)
from utils.coco_util import *
import numpy as np
import torch

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
    tf_image = pil_image_to_tf_image(pil_image,transform)
    # 3.norm_image to denorm image
    denorm_image = norm_image_to_denorm_image(tf_image,mean,std)
    # 4.denorm_image to visual image
    visual_image = tensor_image_to_visual_image(denorm_image)

    plt.imshow(visual_image)
    plt.show()

def transform_images_bboxes():
    img_idx = 150
    img,boxes,labels = get_image_infos(img_idx)
    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    ########## raw image labels ##########
    transformed = trans(image=img, bboxes=boxes)  # labels 是给每个框指定的标签，这里是 0
    # 提取增强后的结果
    transformed_img = transformed['image']  # 处理后的图像
    transformed_bboxes = transformed['bboxes']  # 处理后的边界框

    # 显示结果
    print(f"原始边界框: {boxes}")
    print(f"增强后的边界框: {transformed_bboxes}")

if __name__ == '__main__':
    # test_image_transform()
    transform_images_bboxes()