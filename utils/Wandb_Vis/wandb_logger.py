import wandb

def init_wandb(project_name, entity_name):
    wandb.init(project=project_name, entity=entity_name)

def log_mask(image_name, img, masks, group_labels, group_classes, mask_type):
    wandb_masks = []
    for i, (mask, group_label, group_class) in enumerate(zip(masks, group_labels, group_classes)):
        wandb_masks.append({"mask": {"mask_data": mask, "class_labels": {str(cls): str(cls) for cls in group_class}}, "label": group_label})
    wandb.log({f"{mask_type} Mask": wandb.Image(img, masks=wandb_masks, caption=image_name)})