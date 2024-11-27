# SegFormer 모델 설정
# MIT backbone variants: B0-B5

def get_segformer_config(backbone='b3', img_size=1536, lr=0.00006, max_iters=40000):
    """SegFormer 모델 설정을 반환합니다.
    
    Args:
        backbone (str): 'b0', 'b1', 'b2', 'b3', 'b4', 'b5' 중 선택
        img_size (int): 입력 이미지 크기
        lr (float): Learning rate
        max_iters (int): 최대 학습 반복 횟수
    """
    
    # 백본 설정
    backbone_settings = {
        'b0': {
            'embed_dims': 32,
            'num_heads': [1, 2, 5, 8],
            'num_layers': [2, 2, 2, 2],
            'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'
        },
        'b1': {
            'embed_dims': 64,
            'num_heads': [1, 2, 5, 8],
            'num_layers': [2, 2, 2, 2],
            'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'
        },
        'b2': {
            'embed_dims': 64,
            'num_heads': [1, 2, 5, 8],
            'num_layers': [3, 4, 6, 3],
            'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'
        },
        'b3': {
            'embed_dims': 64,
            'num_heads': [1, 2, 5, 8],
            'num_layers': [3, 4, 18, 3],
            'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'
        },
        'b4': {
            'embed_dims': 64,
            'num_heads': [1, 2, 5, 8],
            'num_layers': [3, 8, 27, 3],
            'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-f6b68236.pth'
        },
        'b5': {
            'embed_dims': 64,
            'num_heads': [1, 2, 5, 8],
            'num_layers': [3, 6, 40, 3],
            'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'
        }
    }
    
    settings = backbone_settings[backbone]
    
    config = {
        'model': {
            'type': 'EncoderDecoderWithoutArgmax',
            'data_preprocessor': {
                'type': 'SegDataPreProcessor',
                'mean': [0., 0., 0.],
                'std': [255., 255., 255.],
                'bgr_to_rgb': True,
                'size': (img_size, img_size),
                'pad_val': 0,
                'seg_pad_val': 255,
            },
            'backbone': {
                'type': 'MixVisionTransformer',
                'init_cfg': {'type': 'Pretrained', 'checkpoint': settings['checkpoint']},
                'embed_dims': settings['embed_dims'],
                'num_heads': settings['num_heads'],
                'num_layers': settings['num_layers']
            },
            'decode_head': {
                'type': 'SegformerHeadWithoutAccuracy',
                'in_channels': [64, 128, 320, 512],
                'num_classes': 29,
                'loss_decode': {
                    'type': 'CrossEntropyLoss',
                    'use_sigmoid': True,
                    'loss_weight': 1.0,
                }
            }
        },
        'optimizer': {
            'type': 'AdamW',
            'lr': lr,
            'betas': (0.9, 0.999),
            'weight_decay': 0.01
        },
        'train_cfg': {
            'type': 'IterBasedTrainLoop',
            'max_iters': max_iters,
            'val_interval': 2000
        },
        'param_scheduler': [
            {
                'type': 'LinearLR',
                'start_factor': 1e-6,
                'by_epoch': False,
                'begin': 0,
                'end': 1500
            },
            {
                'type': 'PolyLR',
                'eta_min': 0.0,
                'power': 1.0,
                'begin': 1500,
                'end': max_iters,
                'by_epoch': False,
            }
        ],
        'fp16': {'loss_scale': 'dynamic'},
        'default_hooks': {
            'timer': {'type': 'IterTimerHook'},
            'logger': {
                'type': 'LoggerHook',
                'interval': 50,
                'log_metric_by_epoch': False
            },
            'param_scheduler': {'type': 'ParamSchedulerHook'},
            'checkpoint': {
                'type': 'CheckpointHook',
                'by_epoch': False,
                'interval': 2000
            },
            'sampler_seed': {'type': 'DistSamplerSeedHook'},
            'visualization': {'type': 'SegVisualizationHook'}
        }
    }
    
    return config

"""
# 사용 예시
1. SegFormer B0 모델 설정 (이미지 크기 2048)
config_b0 = get_segformer_config('b0', img_size=2048)

2. SegFormer B3 모델 설정 (이미지 크기 1536)
config_b3 = get_segformer_config('b3', img_size=1536)

3. SegFormer B5 모델 설정 (이미지 크기 1024)
config_b5 = get_segformer_config('b5' , img_size=1024)
"""