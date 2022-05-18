from pycocotools.coco import COCO
import os
import matplotlib.pyplot as plt
import torch
import json
from PIL import Image
import numpy as np
import tqdm


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, img_path, ann_path, prop_file=None, transform=None):
        '''
        Args:
            dataset_dir: directory of the dataset without file name
            img_path: path to images excluding dataset path
            ann_path: path to annotations file excluding dataset path
            prop: property to classify by
        Return:
            CocoDataset Object
        '''

        ann_file = os.path.join(dataset_dir, ann_path)
        self.imgs_dir = os.path.join(dataset_dir, img_path)
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()

        self.transform = transform
        try:
            fp = open(prop_file)
            self.mapping = json.load(fp)
        except FileNotFoundError:
            print("Property mapping not found")
            self.mapping = None
        except json.decoder.JSONDecodeError:
            print("Invalid JSON file")
            self.mapping = None

    def __getitem__(self, idx):
        '''
        Args:
            idx: index of sample to be fed
        return:
            dict containing:
            - PIL Image of shape (H, W)
            - target (dict) containing:
                - boxes: FloatTensor[N, 4], N being the n° of instances and it's bounding
                boxe coordinates in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H;
                - labels: Int64Tensor[N], class label (0 is background);
                - image_id: Int64Tensor[1], unique id for each image;
                - area: Tensor[N], area of bbox;
                - iscrowd: UInt8Tensor[N], True or False;
                - masks: UInt8Tensor[N, H, W], segmantation maps;
        '''
        img_id = self.img_ids[idx]
        img_obj = self.coco.loadImgs(img_id)[0]
        anns_obj = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        img = Image.open(os.path.join(self.imgs_dir, img_obj['file_name'])).convert('RGB')

        bboxes = [ann['bbox'] for ann in anns_obj]
        masks = [self.coco.annToMask(ann) for ann in anns_obj]
        areas = [ann['area'] for ann in anns_obj]

        boxes = torch.as_tensor(bboxes, dtype=torch.uint8)
        if self.mapping is None:
            labels = []
            for i in anns_obj:
                labels += [i['category_id']]
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            try:
                labels = []
                for i in anns_obj:
                    labels += [self.mapping[str(anns_obj[0]['category_id'])]]
            except IndexError:
                labels += [0]
            labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = torch.as_tensor(areas)
        iscrowd = torch.zeros(len(anns_obj), dtype=torch.int64)

        target = {}
        l = boxes.tolist()
        bboxes = []
        for box in l:
            x, y, w, h = box
            if w == 0 or h == 0:
                continue
            bboxes += [[x, y, x+w, y+h]]
        target["boxes"] = torch.as_tensor(bboxes)
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img = self.transform(img)
        return img, target


    def __len__(self):
        return len(self.img_ids)

def remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10


    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        bbox = [ann['bbox'] for ann in anno]
        if len(bbox) < 1:
            return False
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    ids = []
    for ds_idx, img_id in enumerate(dataset.img_ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset