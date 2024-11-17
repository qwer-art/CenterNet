import os.path as osp
import sys
project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)
from utils.util import *
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

coco = COCO(coco_anno_file)
categories = coco.loadCats(coco.getCatIds())
colors = plt.cm.get_cmap('Dark2', len(categories))
cat_colors = [colors(idx) for idx in range(len(categories))]
cat_ids = [cat['id'] for cat in categories]
cat_names = [cat['name'] for cat in categories]

id2cat = {id: cat for id, cat in zip(cat_ids, cat_names)}
cat2id = {cat: id for id, cat in zip(cat_ids, cat_names)}
id2color = {id: color for id, color in zip(cat_ids, cat_colors)}
cat2color = {cat: color for cat, color in zip(cat_names, cat_colors)}

### (num_images,num_catrgories) (num_objs,)
#### api method
## 1.global infos
def get_global_infos():
    img_ids = list(sorted(coco.imgs.keys()))
    print("number of images: {}".format(len(img_ids)))
    coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
    print("number of classes: {}".format(len(coco_classes)))
## 2. search_by_categories
def search_by_categories(cat_names = ['person','motorbike']):
    print(f"cat_names: {cat_names}")
    catIds = coco.getCatIds(catNms=cat_names)
    print(f"cat_ids: {len(catIds)}")
    imgIds = coco.getImgIds(catIds=catIds)
    print(f"image_ids: {len(imgIds)}")
## 3. seach_by_images: 2009_001960
def search_by_images(img_id = '2009_001960'):
    img_info = coco.loadImgs([img_id])[0]
    print(f"img_id: {img_id},img_info: {img_info}")
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    print(f"ann_ids: {ann_ids}")
    ann_infos = coco.loadAnns(ann_ids)
    for ann_id,ann_info in zip(ann_ids,ann_infos):
        print(f"ann_id: {ann_id},ann_info: {ann_info}")
## 4. draw_image
def draw_image(img_id = '2009_001960'):
    img_info = coco.loadImgs([img_id])[0]
    ann_infos = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))

    img_path = osp.join(dataset_image_path, img_info['file_name'])
    img = Image.open(img_path)

    fig,ax = plt.subplots()
    ax.imshow(img)
    for idx,ann in enumerate(ann_infos):
        ann_id = ann['category_id']
        ann_name = str(idx) + ":" + id2cat[ann_id]
        bbox = ann['bbox']
        ann_color = id2color[ann_id]
        ## bbox
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor=ann_color, facecolor='none')
        ax.add_patch(rect)
        ## center
        cx, cy = bbox[0] + bbox[2] / 2.,bbox[1] + bbox[3] / 2.
        ax.scatter(cx,cy,color=ann_color, marker='x', s=100)
        ## txt
        ax.text(bbox[0],bbox[1],ann_name,fontsize = 12,ha='left', va='top',color='r')

    image_dir = osp.join(project_path,"test")
    image_name = str(img_id) + ".jpg"
    image_path = osp.join(image_dir,image_name)
    # plt.savefig(image_path)
    plt.show()

if __name__ == '__main__':
    get_global_infos()
    # search_by_categories()
    # search_by_images()
    draw_image()
