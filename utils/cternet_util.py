import os.path as osp
import sys

import matplotlib.pyplot as plt

from utils.coco_util import id2cat

project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)
from utils.coco_util import *
import numpy as np
import math

def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2   # 此处没有乘以a1，不过影响不大

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]#np.orgin 生成二维网格坐标

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0 #np.finfo()常用于生成一定格式，数值较小的偏置项eps，以避免分母或对数变量为零
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)  # 逐个元素比较大小，保留大的值
    return heatmap

def on_key(event):
    """ 键盘事件处理函数，按 'n' 键跳到下一张图片 """
    if event.key == 'n':
        plt.close()  # 关闭当前图像，显示下一张图

if __name__ == '__main__':
    num_classes = len(categories)
    img_id = '2009_001960'

    img_info = coco.loadImgs([img_id])[0]
    ann_infos = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))

    hm_height = img_info['height']
    hm_width = img_info['width']
    heat_map = np.zeros((num_classes, hm_height, hm_width), dtype=np.float32)
    print(f"heat_map: {heat_map.shape}")

    image_dir = osp.join(project_path,"test")

    for idx,ann_info in enumerate(ann_infos):
        cat_id = ann_info['category_id']
        bbox = ann_info['bbox']
        id = ann_info['id']
        h,w = bbox[3],bbox[2]
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            ct = np.array(
                [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            one_htmap = draw_umich_gaussian(heat_map[cat_id],ct_int,radius)

            ## image_id-ann_id-cls
            image_name = ann_info['image_id'] + "-" + str(ann_info['id']) + "-" + id2cat[cat_id] + ".jpg"
            image_path = osp.join(image_dir,image_name)

            # 创建一个新的图形
            plt.figure(figsize=(6, 4))  # 设置图像的尺寸，保持原状
            plt.imshow(one_htmap, cmap='viridis', aspect='auto', origin='upper')  # 使用热图显示矩阵
            plt.colorbar()  # 显示颜色条

            # 设置标题等
            image_name = f"{idx}: {id2cat[cat_id]},{ct}"
            plt.title(image_name)
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')

            # 定义按键事件，按 'n' 继续下一张图
            plt.gcf().canvas.mpl_connect('key_press_event', on_key)
            plt.savefig(image_path)
            # 显示图像
            plt.show()