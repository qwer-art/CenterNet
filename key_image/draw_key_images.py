import os.path as osp
import sys
project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)

from utils.coco_util import *
from utils.torch_transform import *
key_image_path = osp.join(project_path,"key_image")

# 1. ann_image
def draw_ann_image(img_idx):
    img_info = get_image_info(img_idx)
    img, boxes, labels = get_image_infos(img_idx)
    fig,ax = plt.subplots()
    ax.imshow(img)
    for idx,(bbox,label) in enumerate(zip(boxes,labels)):
        ann_id = int(label)
        ann_name = str(idx) + ":" + id2cat[ann_id]
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

    # img_idx,len(img_ids),img_id
    img_name = str(img_idx) + '_' + str(get_image_size()) + "_" + img_info['id'] + f"_ann({len(labels)})"+".jpg"
    img_path = osp.join(key_image_path,img_name)

    plt.savefig(img_path)
    plt.show()

# 2. transform_image
def draw_transform_image(img_idx):
    img_info = get_image_info(img_idx)
    img, boxes, labels = get_image_infos(img_idx)
    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    img = transform(img)
    boxes = transform(boxes)
    print(f"img: {img.shape},boxes: {boxes.shape}")

# 3. heatmap_image

if __name__ == '__main__':
    img_idx = 150
    # draw_ann_image(img_idx)
    draw_transform_image(img_idx)