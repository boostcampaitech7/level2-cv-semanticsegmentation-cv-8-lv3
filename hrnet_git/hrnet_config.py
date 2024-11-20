# hrnet_config.py

from yacs.config import CfgNode as CN

def get_hrnet_config(model_width=48):  # 파라미터 추가
    config = CN()
    
    # 기본 설정
    config.MODEL = CN()
    config.MODEL.NAME = 'seg_hrnet'
    config.MODEL.ALIGN_CORNERS = True
    config.MODEL.PRETRAINED = ''
    config.MODEL.EXTRA = CN()
    
    # DATASET 설정
    config.DATASET = CN()
    config.DATASET.NUM_CLASSES = 29
    config.DATASET.DATASET = 'custom'
    
    # MODEL.EXTRA 설정
    config.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
    
    # HRNet width에 따른 채널 수 설정
    if model_width == 48:  # HRNet-W48
        channels = {
            'stage1': [64],
            'stage2': [48, 96],
            'stage3': [48, 96, 192],
            'stage4': [48, 96, 192, 384]
        }
    elif model_width == 64:  # HRNet-W64
        channels = {
            'stage1': [64],
            'stage2': [64, 128],
            'stage3': [64, 128, 256],
            'stage4': [64, 128, 256, 512]
        }
    
    # stage 1
    config.MODEL.EXTRA.STAGE1 = CN()
    config.MODEL.EXTRA.STAGE1.NUM_MODULES = 1
    config.MODEL.EXTRA.STAGE1.NUM_BRANCHES = 1
    config.MODEL.EXTRA.STAGE1.NUM_BLOCKS = [4]
    config.MODEL.EXTRA.STAGE1.NUM_CHANNELS = channels['stage1']
    config.MODEL.EXTRA.STAGE1.BLOCK = 'BOTTLENECK'
    config.MODEL.EXTRA.STAGE1.FUSE_METHOD = 'SUM'

    # stage 2
    config.MODEL.EXTRA.STAGE2 = CN()
    config.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
    config.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
    config.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
    config.MODEL.EXTRA.STAGE2.NUM_CHANNELS = channels['stage2']
    config.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
    config.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

    # stage 3
    config.MODEL.EXTRA.STAGE3 = CN()
    config.MODEL.EXTRA.STAGE3.NUM_MODULES = 4
    config.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
    config.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
    config.MODEL.EXTRA.STAGE3.NUM_CHANNELS = channels['stage3']
    config.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
    config.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

    # stage 4
    config.MODEL.EXTRA.STAGE4 = CN()
    config.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
    config.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
    config.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
    config.MODEL.EXTRA.STAGE4.NUM_CHANNELS = channels['stage4']
    config.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
    config.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

    return config

"""
# 사용 예시
1. HRNet 모델 파일 import를 위한 경로 설정
import sys
sys.path.append('./hrnet_models')  # HRNet 모델 파일이 있는 디렉토리

2. HRNet 모델 가져오기
from seg_hrnet import get_seg_model  # HRNet 모델 생성 함수

3. 설정 가져오기
config = get_hrnet_config()  # 위에서 정의한 함수로 설정 생성

4. 모델 생성
model = get_seg_model(config)  # 설정을 사용해 실제 HRNet 모델 생성


# HRNet-W48 모델 생성
config = get_hrnet_config(model_width=48)
model_w48 = get_seg_model(config)

# HRNet-W64 모델 생성
config = get_hrnet_config(model_width=64)
model_w64 = get_seg_model(config)
"""