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
###
# 1.ann_dict: {dict: 40138}
# 2.catToImgs: {defaultdict: 20} {str: list(str)}
# 3.cats: {dict: 21}
# 4.dataset: {dict: 4}
# 5.imgToAnns: {defaultdict: 17125}
# 6.imgs: {dict: 17125}
# 7.img_dir: str

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
## 5. get_image_info: img_idx
def get_image_size():
    img_ids = list(coco.imgs.keys())
    return len(img_ids)
def get_image_info(img_idx):
    # 1.img_id
    img_ids = list(coco.imgs.keys())
    img_id = [img_ids[img_idx]]
    print(f"img_idx: {img_idx},img_id: {img_id}")
    # 2.image info
    img_info = coco.loadImgs(img_id)[0]
    return img_info
def get_image_anns(img_idx):
    # 1.img_id
    img_ids = list(coco.imgs.keys())
    img_id = [img_ids[img_idx]]
    print(f"img_idx: {img_idx},img_id: {img_id}")
    # 2.image
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(dataset_image_path, img_info['file_name'])
    # 3.ann_ids
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    return anns
def get_image_infos(img_idx):
    # 1.img_info
    img_info = get_image_info(img_idx)
    img_path = os.path.join(dataset_image_path, img_info['file_name'])
    img = Image.open(img_path).convert("RGB")
    # 2.ann_infos
    img_id = [img_info['id']]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    boxes = []
    labels = []
    for ann in anns:
        # COCO中的目标框格式是[x, y, width, height]
        boxes.append(ann['bbox'])
        labels.append(ann['category_id'])
    return img, np.array(boxes), np.array(labels)
def get_image_wh(img_idx):
    img_info = get_image_info(img_idx)
    height = img_info['height']
    width = img_info['width']
    return [width,height]
def get_image_objs(img_idx):
    anns = get_image_anns(img_idx)
    bboxs = []
    clses = []
    for ann in anns:
        bbox = ann['bbox']
        cls = ann['category_id']
        bboxs.append(bbox)
        clses.append(cls)
    return bboxs,clses
def test():
    for id,cat in zip(cat_ids,cat_names):
        print(f"categories: {id},{cat}")

if __name__ == '__main__':
    # get_global_infos()
    # search_by_categories()
    # search_by_images()
    # draw_image()
    test()