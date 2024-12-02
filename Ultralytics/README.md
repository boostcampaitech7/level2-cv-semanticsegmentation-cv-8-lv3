# 🚀 Semantic Segmentation using Ultralytics library

## 🌟 주요 기능

- **Ultralytics YOLOv11**: 최신 Ultralytics 라이브러리를 활용한 Semantic Segmentation 지원

## 🚀 시작하기

### 1️⃣ 데이터 준비
```bash
python3 yolo_dataset_maker.py
```
원본 데이터를 YOLO가 학습할 수 있는 형식의 데이터로 전환해줍니다.

### 2️⃣ 학습 시작
```bash
python3 train.py --config configs/train_config.yaml
```
/configs에 실험에 필요한 인자들을 조정해서 yaml 파일을 만들어서 실험을 진행합니다.

### 3️⃣ 결과 확인
```bash
python3 inference.py
```
학습된 모델을 바탕으로 테스트 이미지를 예측해 결과를 확인합니다.


## 📊 성능
| 모델               | 데이터셋      | Dice Score   | 비고                     |
|--------------------|--------------|--------|--------------------------|
| YOLOv11-seg     | Custom Dataset | 0.9434 | fold1~5 hard voting                |
