import os
import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM

# IMAGE_PATH 설정
IMAGE_PATH = "/data/ephemeral/home/data/val_fold_1/yolo/images"
CHECKPOINT_PATH = "/data/ephemeral/home/kgs/level2-cv-semanticsegmentation-cv-8-lv3/base/checkpoints/last/best_45epoch_0.9704.pt"

# IMAGE_PATH에서 파일 목록 가져오기
def get_random_image(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    random_image_path = os.path.join(image_folder, random.choice(image_files))
    return random_image_path

# 랜덤으로 선택된 이미지 경로
image_path = get_random_image(IMAGE_PATH)

# 이미지 로딩
image = Image.open(image_path)
image = image.resize((256, 256))

# 이미지 채널 확인 및 전처리
if image.mode == 'RGB':
    rgb_img = np.float32(image) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
elif image.mode == 'L':
    rgb_img = np.repeat(np.float32(image)[:, :, np.newaxis], 3, axis=2) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
else:
    raise ValueError("지원되지 않는 이미지 채널 형식입니다.")

# 모델 로드
model = torch.load(CHECKPOINT_PATH)
model = model.eval()

# CUDA 사용 여부 확인
if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = input_tensor.cuda()

# 모델 출력 계산
output = model(input_tensor)

# Target layer 설정 (모델 구조에 따라 수정 필요)
target_layers = [model.decoder.blocks[-2]]

# 클래스 정의
sem_classes = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

# 각 클래스에 대해 Grad-CAM 생성
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()

# Softmax로 각 클래스의 예측값 계산
normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()

all_cam_images = []
for class_category in range(len(sem_classes)):
    anno_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    anno_mask_float = np.float32(anno_mask == class_category)
    targets = [SemanticSegmentationTarget(class_category, anno_mask_float)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        all_cam_images.append(cam_image)

# Grad-CAM 이미지 그리드 정렬
grid_size = (5, 6)
num_classes = len(sem_classes)
assert num_classes <= grid_size[0] * grid_size[1], "Grid size is too small for all classes"

# 여백 및 그리드 설정
margin = 20
title_height = 30
grid_image_width = grid_size[1] * (256 + margin) - margin
grid_image_height = grid_size[0] * (256 + margin + title_height) - margin
grid_image = Image.new('RGB', (grid_image_width, grid_image_height), (0, 0, 0))

# 폰트 설정
try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()

# 각 이미지 배치
for i in range(num_classes):
    row = i // grid_size[1]
    col = i % grid_size[1]
    x = col * (256 + margin)
    y = row * (256 + margin + title_height)

    image = Image.fromarray(all_cam_images[i])
    grid_image.paste(image, (x, y + title_height))

    draw = ImageDraw.Draw(grid_image)
    class_name = sem_classes[i]
    bbox = draw.textbbox((0, 0), class_name, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = x + (256 - text_width) // 2
    text_y = y
    draw.text((text_x, text_y), class_name, fill="white", font=font)

# 그리드 이미지 저장
try:
    grid_image.save("./gradcam_grid_all_classes_decoder.blocks[-2].png")
    print("Grad-CAM grid image saved successfully!")
except Exception as e:
    print(f"Error saving grid image: {e}")
