# 🚀 BASE Semantic Segmentation Project

이 저장소는 **Boostcamp AI Tech 7기**의 Semantic Segmentation 팀 프로젝트로, BASE Semantic Segmentation을 구현하고 실험한 결과를 공유합니다.  
최신 기술과 팀워크를 결합하여 고성능 모델을 개발하고, 이를 활용한 다양한 문제 해결 사례를 탐구합니다.  

## 📂 프로젝트 구조

```
├── configs/         # 학습 및 실험 설정 파일
├── loss/            # 손실 함수 파일
├── models/          # 커스텀 모델 정의 및 설정
├── scheduler/       # 스케쥴러 파일
├── utils/           # 유틸리티 함수 및 도구
├── dataset.py       # 데이터셋 설정 파일
├── inference.py     # 추론 파일
├── README.md        # 프로젝트 소개 문서
├── requirements.txt # 프로젝트 의존성 리스트
├── train.py         # 학습 실행 스크립트
└── trainer.py       # 학습 로직과 루프 관리
```

## 🌟 주요 기능

- **BASE MODEL**: Unet과 efficientnet-b7을 사용하여 기준 Base 모델을 설정
- **커스텀 데이터셋**: 사용자 정의 데이터셋 학습 및 평가
- **모델 성능 향상**: 다양한 기법을 활용한 모델 최적화 및 실험
- **손쉬운 확장**: 코드 구조의 유연성을 통해 추가 기능 및 모델 개발 가능

## 🚀 시작하기

### 1️⃣ 설치
```bash
git clone https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-8-lv3.git
cd level2-cv-semanticsegmentation-cv-8-lv3/base
pip install -r requirements.txt
```

### 2️⃣ 데이터 준비
`datasets/` 디렉토리에 데이터셋을 배치한 후, 적절히 경로를 설정합니다.

### 3️⃣ 학습 시작
```bash
python train.py --config configs/base_train.yaml
```

### 4️⃣ 결과 확인
- `results/` 폴더에서 모델 성능 및 시각화 결과를 확인할 수 있습니다.
## 📊 성능
| Architecture               | Encoder      | LB Score   | 비고                     |
|--------------------|--------------|--------|--------------------------|
| Unet    | efficientnet-b7 | 0.9698 | 기본 설정                |


## 🛠️ 개발 환경
- Python 3.9+
- PyTorch 2.0
