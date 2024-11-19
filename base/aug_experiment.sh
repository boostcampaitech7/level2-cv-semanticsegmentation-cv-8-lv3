#!/bin/bash

# 증강 기법별 YAML 설정 파일 리스트
configs=(
    "configs/light_base.yaml"
    "configs/centercrop.yaml"
    "configs/gaussian_noise.yaml"
    "configs/gaussian_blur.yaml"
    "configs/random_contrast.yaml"
    "configs/random_brightness_contrast.yaml"
    "configs/elastic_transform.yaml"
    "configs/rotate.yaml"
    "configs/geometric_transform.yaml"
)

# 결과 저장 디렉토리 생성
RESULTS_DIR="results"
mkdir -p $RESULTS_DIR

# 각 YAML 파일에 대해 반복 실행
for config in "${configs[@]}"
do
  echo "현재 실행 중인 config: $config"
  # config 파일 이름에서 증강 이름 추출
  AUG_NAME=$(basename "$config" .yaml)
  
  # 학습 실행 및 로그 저장
  python train.py --config $config > "$RESULTS_DIR/${AUG_NAME}.log"
done
