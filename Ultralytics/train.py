import argparse
from ultralytics import YOLO
import wandb

# config 파일 경로를 입력받기 위한 인자 파서 설정
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True, help='config 파일의 경로를 입력하세요')
args = parser.parse_args()

# wandb 초기화
# wandb.init(project="handbone-semantic-segmentation")

model = YOLO("yolo11x-seg.pt")

result = model.train(cfg=args.cfg)
model.val(cfg=args.cfg)