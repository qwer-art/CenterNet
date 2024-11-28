import os.path as osp
import sys
project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)

from utils.coco_util import *
from utils.util import *
from utils.image import *
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import models, transforms
import cv2
import math

class CenterNetDecoder(nn.Module):
    def __init__(self, in_channels, bn_momentum=0.1):
        super(CenterNetDecoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.in_channels = in_channels
        self.deconv_with_bias = False

        # h/32, w/32, 2048 -> h/16, w/16, 256 -> h/8, w/8, 128 -> h/4, w/4, 64
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            num_filter = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=num_filter,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(num_filter, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = num_filter
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)

class CenterNetHead(nn.Module):
    def __init__(self, num_classes=80, channel=64, bn_momentum=0.1):
        super(CenterNetHead, self).__init__()

        # heatmap
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        # bounding boxes height and width
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0))

        # center point offset
        self.offset_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        hm = self.cls_head(x)
        wh = self.wh_head(x)
        offset = self.offset_head(x)

        return hm, wh, offset

# 1. 加载预训练的 ResNet-50 模型
backbone = models.resnet50(pretrained=True)

# 2. 去掉 ResNet-50 的全连接层（fc layer）来获得特征
# 这里我们通过 `torch.nn.Sequential` 保留 ResNet 中的前几层（即卷积和池化层）
backbone = torch.nn.Sequential(*list(backbone.children())[:-2])  # 去掉最后的全连接层

# 3. 定义图像预处理过程
transform = transforms.Compose([
    transforms.Resize(256),  # 调整图像大小
    transforms.CenterCrop(224),  # 中心裁剪
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize(  # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def pred(img):
    img_tensor = transform(img)

    # 增加批次维度
    img_tensor = img_tensor.unsqueeze(0)  # 形状变为 [1, 3, 224, 224]
    print(f"raw_tensor: {img_tensor.shape}")
    # backbone feature map
    backbone_ftmap = backbone(img_tensor)
    print(f"backbone_ftmap: {backbone_ftmap.shape}")
    # decoder feature map
    in_channel = backbone_ftmap.shape[1]
    decoder = CenterNetDecoder(in_channel)
    decoder_ftmap = decoder(backbone_ftmap)
    print(f"decoder_ftmap: {decoder_ftmap.shape}")
    in_channel = decoder_ftmap.shape[1]
    num_classes = len(categories)
    head = CenterNetHead(num_classes,in_channel)
    hm, wh, offset = head(decoder_ftmap)
    print(f"hm: {hm.shape},wh: {wh.shape},offset: {offset.shape}")
    return hm,wh,offset

if __name__ == '__main__':
    img_idx = 150
    img,boxes,labels = get_image_infos(img_idx)
    # pred
    # hm, wh, offset = pred(img)
    # label
    img_wh = img.size
    print(f"img_size: {img_wh}")

