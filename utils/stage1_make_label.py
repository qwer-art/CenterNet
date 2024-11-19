import os.path as osp
import sys
project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)
from utils.coco_util import *

max_objs = 128
def gaussian_radius(bbox_wh, min_overlap=0.7):
    width,height = bbox_wh

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
    return int(min(r1, r2, r3))

def get_centerxy(bbox):
    cx = bbox[0] + bbox[2] / 2.
    cy = bbox[1] + bbox[3] / 2.
    return [int(cx),int(cy)]

def get_center_bound(img_wh,center_xy,radius):
    width,height = img_wh
    cx,cy = center_xy

    left = min(cx,radius)
    right = min(width - cx,radius + 1)
    top = min(cy,radius)
    bottom = min(height - cy,radius + 1)
    return [left,right,top,bottom]

def guassian_map2D(radius,sigma_vs_radius = 1 / 6.):
    m = n = 2 * radius + 1
    sigma = radius * sigma_vs_radius
    y, x = np.ogrid[-m:m + 1, -n:n + 1]#np.orgin 生成二维网格坐标

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0 #np.finfo()常用于生成一定格式，数值较小的偏置项eps，以避免分母或对数变量为零
    return h

def assign_heatmap(ith_hm,guass2d,bound,hm_center,guass_center):
    left,right,top,bottom = bound
    cx,cy = hm_center
    gx,gy = guass_center
    masked_heatmap = ith_hm[cy - top:cy + bottom, cx - left:cx + right]
    masked_gaussian = guass2d[gy - top:gy + bottom, gx - left:gx + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)  # 逐个元素比较大小，保留大的值
    return ith_hm

if __name__ == '__main__':
    img_idx = 150
    ### 1.heatmap
    num_classes = len(categories)
    img_wh = get_image_wh(img_idx)
    bboxes,clses = get_image_objs(img_idx)

    heat_map = np.zeros((num_classes, img_wh[1], img_wh[0]), dtype=np.float32)
    ### input: cls,bbox heatmap:
    for bbox,cls in zip(bboxes,clses):
        # 1. ith_ht = heatmap[cls]
        cls_idx = int(cls)
        # 2. radius = guassian_radius(bbox_wh,min_overlap=0.7): int
        radius = gaussian_radius(bbox[2:])
        # 3. int_center_xy = get_centerxy(bbox_wh): [int,int]
        int_center_xy = get_centerxy(bbox)
        # 4. bound = get_center_bound(img_wh,center_xy,radius): [left,right,top,bottom]
        bound = get_center_bound(img_wh,int_center_xy,radius)
        # 5. guass2d = guassian_map2D(radius,sigma_vs_radius = 1 / 6.): matrix
        guass2d = guassian_map2D(radius)
        # 6. assign: ith_hm = assign_heatmap(ith_hm,guass2d,bound)
        heat_map[cls_idx] = assign_heatmap(heat_map[cls_idx], guass2d, bound, int_center_xy, [radius,radius])

        plt.figure(figsize=(6, 4))  # 设置图像的尺寸，保持原状
        plt.imshow(heat_map[cls_idx] , cmap='viridis', aspect='auto', origin='upper')  # 使用热图显示矩阵
        plt.colorbar()  # 显示颜色条
        plt.show()
