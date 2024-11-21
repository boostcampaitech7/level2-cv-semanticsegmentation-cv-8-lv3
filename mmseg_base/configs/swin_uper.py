# Train Segformer Mit B3
_base_ = [
    "../mmsegmentation/configs/_base_/models/upernet_swin.py",
    "../xray.py",
    "../mmsegmentation/configs/_base_/default_runtime.py",
    "../mmsegmentation/configs/_base_/schedules/schedule_160k.py",
]
checkpoint_file = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth"
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[0.0, 0.0, 0.0],
    std=[255.0, 255.0, 255.0],
    bgr_to_rgb=True,
    size=(512, 512),
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type="EncoderDecoderWithoutArgmax",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
    ),
    decode_head=dict(
        type="UPerHeadWithoutAccuracy",
        in_channels=[96, 192, 384, 768],
        num_classes=29,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=dict(
        type="FCNHeadWithoutAccuracy",
        in_channels=384,
        num_classes=29,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            loss_weight=1.0,
        ),
    ),
)

## optimizer
# optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# mixed precision
fp16 = dict(loss_scale="dynamic")

# learning policy
# param_scheduler = [
#    dict(
#        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500
#    ),
#    dict(
#        type='PolyLR',
#        eta_min=0.0,
#        power=1.0,
#        begin=1500,
#        end=20000,
#        by_epoch=False,
#    )
# ]
# training schedule for 20k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=20000, val_interval=2000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=2000),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="WandbVisBackend"),
]
visualizer = dict(type="SegLocalVisualizer", vis_backends=vis_backends)
