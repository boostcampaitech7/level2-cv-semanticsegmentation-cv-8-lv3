import os
import wandb
import dotenv
import yaml
import numpy as np
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
dataset = XRayDataset(image_root=image_root, label_root=label_root, is_train=False, fold=0)
visual_loader = DataLoader(dataset, batch_size=1)

# CSV에서 마스크 생성
mask_dict = create_pred_mask_dict(csv_path, 512)

# 시각화 루프 (기존 코드와 동일)
for image_names, images, labels in visual_loader:
    for image_name, image, label in zip(image_names, images, labels):
        print(f"Uploading {image_name}...")
        img, gt = ready_for_visualize(image, label)
        pred = mask_dict.get(image_name, None)

        combined_mask_gt = np.zeros((len(CLASS_GROUPS), 512, 512), dtype=np.uint8)
        combined_mask_pred = np.zeros((len(CLASS_GROUPS), 512, 512), dtype=np.uint8)
        combined_mask_cmp = np.zeros((len(CLASS_GROUPS), 512, 512), dtype=np.uint8)

        # 벡터화된 마스크 생성 (기존 코드와 동일)
        for group_id, group_classes in enumerate(CLASS_GROUPS):
            gt_masks = gt[np.array(group_classes) - 1]
            if pred:
                pred_masks = np.stack([pred.get(CLASSES[class_index - 1], np.zeros((512, 512), dtype=np.uint8)) for class_index in group_classes])
                combined_mask_cmp[group_id] = np.select(
                    [
                        (gt_masks.sum(axis=0) > 0) & (pred_masks.sum(axis=0) == 0),
                        (gt_masks.sum(axis=0) == 0) & (pred_masks.sum(axis=0) > 0),
                        (gt_masks.sum(axis=0) > 0) & (pred_masks.sum(axis=0) > 0)
                    ],
                    [1, 2, 3], 0
                )
                combined_mask_pred[group_id] = np.max(np.where(pred_masks > 0, np.array(group_classes)[:, None, None], 0), axis=0)
            combined_mask_gt[group_id] = np.max(np.where(gt_masks > 0, np.array(group_classes)[:, None, None], 0), axis=0)

        # wandb 로그 기록 (간소화된 코드)
        masks_gt = [{'mask': {'mask_data': m, 'class_labels': {str(c): str(c) for c in g}}, 'label': l} for m, l, g in zip(combined_mask_gt, CLASS_GROUP_LABEL, CLASS_GROUPS)]
        masks_pred = [{'mask': {'mask_data': m, 'class_labels': {str(c): str(c) for c in g}}, 'label': l} for m, l, g in zip(combined_mask_pred, CLASS_GROUP_LABEL, CLASS_GROUPS)]
        masks_cmp = [{'mask': {'mask_data': m, 'class_labels': {str(c): str(c) for c in g}}, 'label': l} for m, l, g in zip(combined_mask_cmp, CLASS_GROUP_LABEL, [{1: "GT Only", 2: "Pred Only", 3: "Overlap"} for _ in CLASS_GROUPS])]


        wandb.log({"GT Mask": wandb.Image(img, masks=masks_gt)})
        wandb.log({"Pred Mask": wandb.Image(img, masks=masks_pred)})
        wandb.log({"Comparison Mask": wandb.Image(img, masks=masks_cmp)})

wandb.finish()