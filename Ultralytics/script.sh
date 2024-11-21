#!/bin/bash

# for i in {1..5}
# do
#     echo "Fold $i 학습을 시작합니다..."
#     python3 train.py --cfg ./configs/base_fold${i}.yaml
#     echo "Fold $i 학습이 완료되었습니다."
# done

# 각 폴드별로 학습 및 추론 실행
for i in {1..1}
do
    echo "Fold $i 학습을 시작합니다..."
    python3 train.py --cfg ./configs/imgsz2048.yaml
    echo "Fold $i 학습이 완료되었습니다."

    echo "Fold $i 모델로 추론을 시작합니다..."
    python3 inference.py \
        --model_path ./handbone-segmentation/imgsz2048/weights/best.pt \
        --save_path ./imgsz2048.csv
    echo "Fold $i 추론이 완료되었습니다."
done

