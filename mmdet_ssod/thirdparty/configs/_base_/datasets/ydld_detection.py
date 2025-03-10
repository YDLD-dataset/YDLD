# dataset settings
dataset_type = 'YDLDDataset'

data_root = "data/YDLD"


img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[58.98896478, 54.15064995, 54.52128607], std=[58.75798357, 53.93865391, 54.30785936], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


self_pipeline = [
    dict(type='LoadImageFromWebcam'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

YDLD_Train = dict(
    type=dataset_type,
    ann_file=data_root + 'annotations/train.json',
    img_prefix=data_root + 'Train/',
    pipeline=train_pipeline
)

YDLD_test = dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'Val/',
        pipeline=test_pipeline)


self_learning = dict(
    type=dataset_type,
    pipeline=self_pipeline
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=YDLD_Train,
    val = YDLD_test,
    test = YDLD_test,
)

evaluation = dict(interval=1, metric='bbox')
