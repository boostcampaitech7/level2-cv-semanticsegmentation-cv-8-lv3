# 🚀 MMSegmentation Custom Semantic Segmentation Project

이 저장소는 **Boostcamp AI Tech 7기**의 Semantic Segmentation 팀 프로젝트로, MMSegmentation 기반의 커스텀 Semantic Segmentation을 구현하고 실험한 결과를 공유합니다.
최신 기술과 팀워크를 결합하여 고성능 모델을 개발하고, 이를 활용한 다양한 문제 해결 사례를 탐구합니다.

## 📂 프로젝트 구조
```
├── datasets/      # 데이터셋 관련 스크립트 및 설정 파일
├── models/        # 커스텀 모델 정의 및 설정
├── configs/       # 학습 및 실험 설정 파일
├── utils/         # 유틸리티 함수 및 도구
├── results/       # 실험 결과 및 시각화
└── README.md      # 프로젝트 소개 문서
```

## 🌟 주요 기능
- **MMSegmentation SegFormer**: 최신 MMSegmentation 라이브러리를 활용한 Semantic Segmentation 지원
- **커스텀 데이터셋**: 사용자 정의 데이터셋 학습 및 평가
- **모델 성능 향상**: 다양한 기법을 활용한 모델 최적화 및 실험
- **손쉬운 확장**: 코드 구조의 유연성을 통해 추가 기능 및 모델 개발 가능

## 🚀 시작하기

### 1️⃣ 설치
```bash
git clone https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-8-lv3.git
cd level2-cv-semanticsegmentation-cv-8-lv3/Ultralytics
pip install -r requirements.txt
```

### 2️⃣ 데이터 준비
`datasets/` 디렉토리에 데이터셋을 배치한 후, 적절히 경로를 설정합니다.

### 3️⃣ 학습 시작
```bash
python train.py --config configs/train_config.yaml
```

### 4️⃣ 결과 확인
- `results/` 폴더에서 모델 성능 및 시각화 결과를 확인할 수 있습니다.

## 📊 성능
| Architecture               | Encoder      | LB Score   | 비고                     |
|--------------------|--------------|--------|--------------------------|
| SegFormer    | Mit-B3 (base) | 0.9451  | Input 이미지 사이즈 = 512x512               |
| SegFormer    | Mit-B4 | 0.9598 | input 이미지 사이즈 = 1024x1024          |
| SegFormer    | Mit-B3  | 0.9685 | input 이미지 사이즈 = 1536x1536              |
| SegFormer    | Mit-B3  | 0.9370 | input 이미지 사이즈 = 1536x1536, inference 이미지 사이즈: 2048x2048                |
| SegFormer    | Mit-B0  | 0.9584  | input 이미지 사이즈 = 2048x2048               |
| SegFormer    | Mit-B5  | 0.9662  | input 이미지 사이즈 = 1024x1024               |


## 🔧 개발 환경
- Python 3.9+
- PyTorch 2.0
- MMSegmentation SegFormer
