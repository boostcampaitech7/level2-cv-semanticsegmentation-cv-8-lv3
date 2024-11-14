import argparse
from ultralytics import YOLO

# config 파일 경로를 입력받기 위한 인자 파서 설정
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True, help='config 파일의 경로를 입력하세요')
args = parser.parse_args()

model = YOLO("yolo11x-seg.pt")

model.train(cfg=args.cfg)

