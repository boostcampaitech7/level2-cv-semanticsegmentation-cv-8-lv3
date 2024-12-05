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
    "finger-1",
    "finger-2",
    "finger-3",
    "finger-4",
    "finger-5",
    "finger-6",
    "finger-7",
    "finger-8",
    "finger-9",
    "finger-10",
    "finger-11",
    "finger-12",
    "finger-13",
    "finger-14",
    "finger-15",
    "finger-16",
    "finger-17",
    "finger-18",
    "finger-19",
    "Trapezium",
    "Trapezoid",
    "Capitate",
    "Hamate",
    "Scaphoid",
    "Lunate",
    "Triquetrum",
    "Pisiform",
    "Radius",
    "Ulna",
]


class XRayDataset(Dataset):
    def __init__(
        self,
        fnames,
        labels,
        image_root,
        label_root,
        fold=0,
        transforms=None,
        is_train=True,
        label_slice=None,
    ):
        self.transforms = A.Compose(transforms)
        self.is_train = is_train
        self.image_root = image_root
        self.label_root = label_root
        self.validation_fold = fold
        self.class2ind = {v: i for i, v in enumerate(CLASSES)}
        self.ind2class = {v: k for k, v in self.class2ind.items()}
        self.num_classes = len(CLASSES)
        self.label_slice = label_slice

        groups = [osp.dirname(fname) for fname in fnames]

        # dummy label
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

        if self.label_slice is not None:
            self.num_classes = len(self.label_slice)
            sliced_classes = [CLASSES[i] for i in self.label_slice]
            self.class2ind = {i: v for i, v in zip(sliced_classes, self.label_slice)}
            self.ind2class = {v: i for i, v in zip(sliced_classes, self.label_slice)}

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, item):
        image_name = self.fnames[item]
        image_path = osp.join(self.image_root, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        label_name = self.labels[item]
        label_path = osp.join(self.label_root, label_name)

        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(CLASSES),)
        label = np.zeros(label_shape, dtype=np.uint8)

        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        if self.label_slice is not None:
            annotations = [
                ann for ann in annotations if ann["label"] in self.class2ind.keys()
            ]

        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = self.class2ind[c]
            points = np.array(ann["points"])

            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.transforms is not None:
            inputs = (
                {"image": image, "mask": label} if self.is_train else {"image": image}
            )
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  # channel first 포맷으로 변경합니다.
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        if self.label_slice is not None:
            label = label[self.label_slice, :, :]

        return image, label


class XRayInferenceDataset(Dataset):
    def __init__(self, fnames, image_root, transforms=None, label_slice=None):
        self.fnames = np.array(sorted(fnames))
        self.image_root = image_root
        self.transforms = transforms
        self.label_slice = label_slice
        self.ind2class = {i: v for i, v in enumerate(CLASSES)}

        if self.label_slice is not None:
            print(self.label_slice)
            classes = [CLASSES[i] for i in self.label_slice]
            self.ind2class = {i: v for i, v in zip(self.label_slice, classes)}

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, item):
        image_name = self.fnames[item]
        image_path = osp.join(self.image_root, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()

        return image, image_name
