import os
import cv2
import json
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

class XRayDataset(Dataset):
    def __init__(self, image_root, label_root, transforms=None, is_train=True):
        self.transforms = A.Compose(transforms) if transforms else None
        self.is_train = is_train
        self.image_root = image_root
        self.label_root = label_root
        self.class2ind = {v: i for i, v in enumerate(CLASSES)}
        self.num_classes = len(CLASSES)

        self.image_ids = sorted([f.split('.')[0] for f in os.listdir(image_root) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.labels = sorted([f.split('.')[0] for f in os.listdir(label_root) if f.endswith('.json')])

        if set(self.image_ids) != set(self.labels):
            print("Warning: Image and Label IDs do not match!")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_id = self.image_ids[item]
        image_path = os.path.join(self.image_root, f"{image_id}.png")
        label_path = os.path.join(self.label_root, f"{image_id}.json")
        print(f"label_path: {label_path}")

        try:
            if not os.path.isfile(image_path) or not os.path.isfile(label_path):
                raise FileNotFoundError(f"Missing file for ID {image_id}")

            image = cv2.imread(image_path)
            image = image / 255.0
            label_shape = tuple(image.shape[:2]) + (self.num_classes,)
            label = np.zeros(label_shape, dtype=np.uint8)

            with open(label_path, "r") as f:
                annotations = json.load(f).get("annotations", [])
                for ann in annotations:
                    class_ind = self.class2ind.get(ann["label"], None)
                    if class_ind is None:
                        continue
                    points = np.array(ann["points"])

                    # points의 차원 확인 및 처리
                    if points.ndim == 3:  # points가 (n, 1, 2) 형태라면 그대로 사용
                        class_label = np.zeros(image.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(class_label, [points], 1)
                    else:  # points의 차원이 잘못되었을 때 (예: (N, 2))
                        points = points.reshape((-1, 1, 2))
                        class_label = np.zeros(image.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(class_label, [points], 1)
                    label[..., class_ind] = class_label

            if self.transforms:
                inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
                result = self.transforms(**inputs)
                image = result["image"]
                label = result["mask"] if self.is_train else label

            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            label = torch.from_numpy(label.transpose(2, 0, 1)).float()
            return image_id, image, label

        except Exception as e:
            print(f"Error for ID {image_id}: {e}")
            default_image = torch.zeros((3, 512, 512), dtype=torch.float32)
            default_label = torch.zeros((self.num_classes, 512, 512), dtype=torch.float32)
            return image_id, default_image, default_label