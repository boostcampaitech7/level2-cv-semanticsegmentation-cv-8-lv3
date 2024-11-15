#!/bin/bash

for i in {1..5}
do
    echo "Fold $i 학습을 시작합니다..."
    python3 train.py --cfg ./configs/base_fold${i}.yaml
    echo "Fold $i 학습이 완료되었습니다."
done
