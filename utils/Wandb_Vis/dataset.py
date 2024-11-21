import os
import cv2
import json
import torch
import numpy as np
import os.path as osp
import albumentations as A

from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]


class XRayDataset(Dataset):
    def __init__(self,  image_root, label_root, fold=0, transforms=None, is_train=True):
        self.transforms = A.Compose(transforms)
        self.is_train = is_train
        self.image_root = image_root
        self.label_root = label_root
        self.validation_fold = fold
        self.class2ind = {v: i for i, v in enumerate(CLASSES)}
        self.ind2class = {v: k for k, v in self.class2ind.items()}
        self.num_classes = len(CLASSES)

        # 파일 이름 목록을 여기서 생성하도록 변경 (메모리 절약)
        fnames = sorted([f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]) # 확장자에 맞춰 수정
        labels = sorted([f for f in os.listdir(label_root) if f.endswith('.json')]) # json 확장자

        groups = [osp.dirname(fname) for fname in fnames]
        ys = [0] * len(fnames)
        gkf = GroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(fnames, ys, groups)):
            if self.is_train:
                if i == self.validation_fold:
                    continue        
                filenames += list(fnames[y])
                labelnames += list(labels[y])
            else:
                if i != self.validation_fold:
                    continue
                filenames = list(fnames[y])
                labelnames = list(labels[y])
                break

        self.fnames = filenames
        self.labels = labelnames


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, item):
        image_name = self.fnames[item]
        label_name = self.labels[item]

        image_path = osp.join(self.image_root, image_name)
        label_path = osp.join(self.label_root, label_name)

        image = cv2.imread(image_path)
        image = image / 255.  # 이미지 정규화

        label_shape = tuple(image.shape[:2]) + (len(CLASSES),)
        label = np.zeros(label_shape, dtype=np.uint8)

        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        for ann in annotations:
            c = ann["label"]
            class_ind = self.class2ind[c]
            points = np.array(ann["points"])
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"] if self.is_train else label

        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label