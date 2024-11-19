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
# MODEL PATH 설정
CHECKPOINT_PATH = "/data/ephemeral/home/jseo/level2-cv-semanticsegmentation-cv-8-lv3/base/checkpoints/Unet_crop/best_35epoch_0.8005.pt"

# IMAGE_PATH에서 파일 목록 가져오기
def get_random_image(image_folder):
    # 폴더 내의 모든 파일을 리스트로 가져오기 (서브폴더는 제외)
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    # 랜덤으로 하나의 이미지 선택
    random_image_path = os.path.join(image_folder, random.choice(image_files))
    return random_image_path

# 랜덤으로 선택된 이미지 경로
image_path = get_random_image(IMAGE_PATH)

# 이미지 로딩
image = Image.open(image_path)
image = image.resize((256, 256))

# 이미지 채널 확인
if image.mode == 'RGB':
    print(f"이미지 채널: RGB (3채널)")
    rgb_img = np.float32(image) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
elif image.mode == 'L':  # 그레이스케일 이미지인 경우
    print(f"이미지 채널: Grayscale (1채널)")
    rgb_img = np.repeat(np.float32(image)[:, :, np.newaxis], 3, axis=2) / 255  # 1채널 -> 3채널로 변환
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
else:
    raise ValueError("지원되지 않는 이미지 채널 형식입니다.")

# 모델 로드
model = torch.load(CHECKPOINT_PATH)  # 모델 전체를 로드
model = model.eval()

# CUDA 사용 여부 확인
if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = input_tensor.cuda()

# 모델 출력
output = model(input_tensor)

# Model 구조에 따라 target layer 설정 다르게 줘야함. 현재는 UNet 기준
target_layers = [model.decoder.blocks[-1]]

# 손목 뼈 X-ray 데이터셋의 클래스 정의
sem_classes = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

# 클래스 인덱스 매핑
sem_class_to_idx = {cls: idx for idx, cls in enumerate(sem_classes)}

# 원하는 클래스 (예: Scaphoid) 선택
class_category = sem_class_to_idx["Scaphoid"]
print(f"scaphoid : {class_category}")

# 예시 모델 출력 텐서 (output은 모델의 raw output)
normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()

# 각 클래스의 마스크를 시각화합니다.
num_classes = len(sem_classes)
fig, axes = plt.subplots(1, num_classes, figsize=(20, 5))

for i in range(num_classes):
    mask = normalized_masks[0, i, :, :].detach().cpu().numpy()
    axes[i].imshow(mask, cmap='gray')
    axes[i].set_title(sem_classes[i])

plt.show()

# 예시 이미지 (image는 PIL 이미지 객체일 수 있음)
# 이미지가 PIL.Image 객체일 경우 numpy 배열로 변환
image = np.array(image)  # PIL 이미지 -> numpy 배열로 변환

# Scaphoid 클래스에 대한 마스크 생성 (예시 normalized_masks)
anno_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
anno_mask_uint8 = 255 * np.uint8(anno_mask == class_category)
anno_mask_float = np.float32(anno_mask == class_category)

# image가 흑백 이미지일 경우, 3D 형태로 변환 (height, width -> height, width, 3)
if len(image.shape) == 2:
    image = np.repeat(image[:, :, None], 3, axis=-1)  # 3채널로 확장

# Scaphoid 마스크를 RGB로 확장 (height, width -> height, width, 3)
anno_mask_uint8_rgb = np.repeat(anno_mask_uint8[:, :, None], 3, axis=-1)

# 이미지와 마스크 결합 (수평 결합)
both_images = np.hstack((image, anno_mask_uint8_rgb))


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()
    
targets = [SemanticSegmentationTarget(class_category, anno_mask_float)]

with GradCAM(model=model,
             target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Generate Grad-CAM for all classes
all_cam_images = []
for class_category in range(len(sem_classes)):
    anno_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    anno_mask_float = np.float32(anno_mask == class_category)
    targets = [SemanticSegmentationTarget(class_category, anno_mask_float)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        all_cam_images.append(cam_image)


# Arrange Grad-CAM images in a grid
grid_size = (5, 6)
num_classes = len(sem_classes)
assert num_classes <= grid_size[0] * grid_size[1], "Grid size is too small for all classes"

# 여백 크기 설정
margin = 20  # 이미지 사이의 여백 크기 (픽셀 단위)
title_height = 30  # 타이틀 텍스트 높이

# 여백과 타이틀 높이를 고려한 전체 그리드 이미지 크기 계산
grid_image_width = grid_size[1] * (256 + margin) - margin  # 마지막 이미지에는 추가 여백이 필요 없음
grid_image_height = grid_size[0] * (256 + margin + title_height) - margin  # 이미지 위 타이틀 높이 포함

# 전체 그리드 이미지 생성
grid_image = Image.new('RGB', (grid_image_width, grid_image_height), (0, 0, 0))  # 검은색 배경

# 폰트 로드 (기본 폰트 사용)
try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()

# 각 이미지와 타이틀 배치
for i in range(num_classes):
    row = i // grid_size[1]
    col = i % grid_size[1]
    x = col * (256 + margin)  # 각 이미지의 x 위치 (여백 포함)
    y = row * (256 + margin + title_height)  # 각 이미지의 y 위치 (여백 및 타이틀 높이 포함)

    # 이미지를 PIL 이미지로 변환하고 그리드 이미지에 붙이기
    image = Image.fromarray(all_cam_images[i])
    grid_image.paste(image, (x, y + title_height))  # 타이틀 아래에 이미지 붙이기

    # 타이틀 텍스트 그리기
    draw = ImageDraw.Draw(grid_image)
    class_name = sem_classes[i]
    bbox = draw.textbbox((0, 0), class_name, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = x + (256 - text_width) // 2  # 텍스트를 이미지 중앙에 맞추기
    text_y = y  # 타이틀 텍스트의 y 위치
    draw.text((text_x, text_y), class_name, fill="white", font=font)  # 흰색 텍스트로 타이틀 그리기

# 그리드 이미지 저장
try:
    grid_image.save("./gradcam_grid_all_classes_with_titles_and_margin.png")
    print("Grad-CAM grid image with titles and margins saved successfully!")
except Exception as e:
    print(f"An error occurred while saving the grid image: {e}")
