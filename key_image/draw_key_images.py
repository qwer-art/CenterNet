import os.path as osp
import sys
project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)

from utils.coco_util import *
key_image_path = osp.join(project_path,"key_image")

# 1. ann_image
# 2. transform_image
# 3. heatmap_image

if __name__ == '__main__':
    img_idx = 150
