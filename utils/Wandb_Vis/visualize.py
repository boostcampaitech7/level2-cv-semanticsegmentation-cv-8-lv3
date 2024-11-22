import os
import cv2
import wandb
import dotenv
import yaml
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
from dataset import XRayDataset
from utils.image_utils import ready_for_visualize
from utils.rle_utils import create_pred_mask_dict
from utils.constants import CLASS_GROUPS, CLASS_GROUP_LABEL, CLASSES

dotenv.load_dotenv()

def set_wandb(configs):
    wandb.login(key=os.getenv(configs['api_key']))
    wandb.init(
        entity=configs['team_name'],
        project=configs['project_name'],
        name=configs['experiment_detail'],
    )

# YAML 설정 파일 경로
config_path = "/data/ephemeral/home/hwcho/level2-cv-semanticsegmentation-cv-8-lv3/utils/Wandb_Vis/config.yaml"

# YAML 파일 로드
with open(config_path, 'r') as file:
    configs = yaml.safe_load(file)

# WandB 설정
set_wandb(configs)

# 데이터셋 경로 설정 (여러분의 데이터 경로로 변경)
image_root = "/data/ephemeral/home/data/val_fold_1/DCM"
label_root = "/data/ephemeral/home/data/val_fold_1/outputs_json"
csv_path = "/data/ephemeral/home/hwcho/val.csv"

# 데이터셋 및 DataLoader 설정
dataset = XRayDataset(image_root=image_root, label_root=label_root, is_train=False, transforms=A.Compose([A.Resize(height=2048, width=2048)])) # Wrap transform in a list
visual_loader = DataLoader(dataset, batch_size=1)

# CSV에서 마스크 생성
mask_dict = create_pred_mask_dict(csv_path, 2048)

class_labels = [{} for _ in range(len(CLASS_GROUPS))]
for idx, class_group in enumerate(CLASS_GROUPS):
    for class_idx in class_group:
        class_labels[idx][class_idx]=CLASSES[class_idx-1] 
        
# 시각화 루프 (기존 코드와 동일)
for image_names, images, labels in visual_loader:
    for image_name, image, label in zip(image_names, images, labels):
        print(f"Uploading {image_name}...")
        img, gt = ready_for_visualize(image, label)
        image_name_with_extension = image_name + '.png'
        pred = mask_dict.get(image_name_with_extension, None)

        combined_mask_gt = np.zeros((len(CLASS_GROUPS), 2048, 2048), dtype=np.uint8)
        combined_mask_pred = np.zeros((len(CLASS_GROUPS), 2048, 2048), dtype=np.uint8)
        combined_mask_cmp = np.zeros((len(CLASS_GROUPS), 2048, 2048), dtype=np.uint8)

        # 벡터화된 마스크 생성 (수정된 부분)
        for group_id, group_classes in enumerate(CLASS_GROUPS):
            gt_masks = gt[np.array(group_classes) - 1]
            if pred:
                # Resize all gt_masks and pred_masks to (2048, 2048)
                gt_masks_resized = np.array([cv2.resize(mask, (2048, 2048)) for mask in gt_masks])
                pred_masks_resized = np.array([pred.get(CLASSES[class_index - 1], np.zeros((2048, 2048), dtype=np.uint8)) for class_index in group_classes])

                # 각 경우별 마스크 생성
                gt_only = np.zeros((2048, 2048), dtype=np.uint8)
                pred_only = np.zeros((2048, 2048), dtype=np.uint8)
                overlap = np.zeros((2048, 2048), dtype=np.uint8)

                for mask_gt, mask_pred in zip(gt_masks_resized, pred_masks_resized):
                    # GT만 존재
                    gt_only += ((mask_gt > 0) & (mask_pred == 0)).astype(np.uint8)
                    # Pred만 존재
                    pred_only += ((mask_gt == 0) & (mask_pred > 0)).astype(np.uint8)
                    # Overlap
                    overlap += ((mask_gt > 0) & (mask_pred > 0)).astype(np.uint8)

                # GT, Pred, Overlap을 하나의 마스크로 합침
                combined_mask_cmp[group_id] = (
                    gt_only * 1 +  # GT는 1로 표시
                    pred_only * 2 +  # Pred는 2로 표시
                    overlap * 3  # Overlap은 3으로 표시
                )
                
                max_result = np.max(np.where(pred_masks_resized > 0, np.array(group_classes)[:, None, None], 0), axis=0)
                combined_mask_pred[group_id] = max_result
                
            max_result_gt = np.max(np.where(gt_masks_resized > 0, np.array(group_classes)[:, None, None], 0), axis=0)
            if max_result_gt.dtype != np.uint8:
                max_result_gt = max_result_gt.astype(np.uint8)  # uint8로 변환

            combined_mask_gt[group_id] = max_result_gt

        # Mask Compare의 클래스 레이블 업데이트
        class_labels_cmp = [{1: "GT", 2: "Pred", 3: "Overlap"} for _ in range(len(CLASS_GROUPS))]

        masks_gt = dict()
        for i, mask in enumerate(combined_mask_gt):
            masks_gt[CLASS_GROUP_LABEL[i]] = dict(mask_data=mask,class_labels=class_labels[i])
        masked_image_gt = wandb.Image(img, masks=masks_gt, caption=image_name)      
        wandb.log({"GT Mask":masked_image_gt})

        masks_pred = dict()
        for i, mask in enumerate(combined_mask_pred):
            masks_pred[CLASS_GROUP_LABEL[i]] = dict(mask_data=mask,class_labels=class_labels[i])
        masked_image_pred = wandb.Image(img, masks=masks_pred, caption=image_name)  
        wandb.log({"Pred Mask":masked_image_pred})   

        masks_cmp = dict()
        for i, mask in enumerate(combined_mask_cmp):
            masks_cmp[CLASS_GROUP_LABEL[i]] = dict(mask_data=mask,class_labels=class_labels_cmp[i])
        masked_image_cmp = wandb.Image(img, masks=masks_cmp, caption=image_name)      
        wandb.log({"Mask Compare":masked_image_cmp})

wandb.finish()