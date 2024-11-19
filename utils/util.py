import os
import os.path as osp
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re
import torchvision.transforms as transforms
import albumentations as A

voc_path = "/home/zyt/Data/VOCdevkit"
voc_ann_path = osp.join(voc_path,"VOC2012/Annotations")
dataset_image_path = osp.join(voc_path, "VOC2012/JPEGImages")
coco_path = "/home/zyt/Data/CocoData"
coco_anno_file = osp.join(coco_path,"annotation.json")
proj_root_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
test_path = osp.join(proj_root_path,'test')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.Resize(512),
    transforms.CenterCrop(512)
])
trans = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # 随机应用仿射变换：平移，缩放和旋转输入
        A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
    ])

### get_vocxml_pathes
def get_annpathes(directory):
    xml_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            xml_files.append(osp.join(directory,filename))
    return xml_files

### get_lables
def parser_label2id(xml_files):
    labels = set()
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # 遍历 <object> 标签
        for obj in root.findall('object'):
            # 获取 <name> 标签，即物体的类别
            label = obj.find('name').text
            labels.add(label)
    labels = ['unknow'] + list(labels)
    label2id = {label : idx for idx,label in enumerate(labels)}
    return labels,label2id

### get_image_info
def get_image_info_from_root(ann_root):
    filename = ann_root.findtext("filename")
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]

    size = ann_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))
    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info

### get_ann_infos
def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann

### convert voc to coco
def convert_voc2coco(input_voc_ann,output_coco_file):
    xml_files = get_annpathes(input_voc_ann)
    labels, label2id = parser_label2id(xml_files)

    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    for a_path in tqdm(xml_files):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()
        # image_info
        img_info = get_image_info_from_root(ann_root)
        output_json_dict['images'].append(img_info)
        img_id = img_info['id']
        # ann_infos
        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj, label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_coco_file, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)

def test(input_voc_ann,output_coco_file):
    xml_files = get_annpathes(input_voc_ann)
    labels, label2id = parser_label2id(xml_files)
    for a_path in tqdm(xml_files):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        num_obj = list()
        num_class = set()
        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj, label2id)
            num_obj.append(ann['category_id'])
            num_class.add(ann['category_id'])
        if len(num_class) >= 2 and len(num_obj) > len(num_class):
            filename = osp.splitext(osp.basename(a_path))[0]
            print(f"filename: {filename}")


if __name__ == '__main__':
    input_voc_ann = voc_ann_path
    output_coco_file = coco_anno_file
    # convert_voc2coco(input_voc_ann, output_coco_file)
    test(input_voc_ann, output_coco_file)