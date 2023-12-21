# dataset settings
dataset_type = 'BDDTrackingDataset'
data_root = 'data/bdd100k/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqResize',
        img_scale=(1920, 1080),  # necessary since few sequences have different img shape
        share_params=True,
        keep_ratio=False,
        bbox_clip_border=False),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
            'gt_instance_ids'
        ]),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1088, 1088),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root +
            'annotations/box_track_20/box_track_val_cocofmt.json',
            img_prefix=data_root + 'images/track/val/',
            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(num_ref_imgs=1, frame_range=3, method='uniform'),
            pipeline=train_pipeline,
            classes=['pedestrian']),
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/box_track_20/box_track_val_cocofmt.json',
        img_prefix=data_root + 'images/track/val/',
        ref_img_sampler=None,
        pipeline=test_pipeline,
        classes=['pedestrian']),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/box_track_20/box_track_val_cocofmt.json',
        img_prefix=data_root + 'images/track/val/',
        ref_img_sampler=None,
        pipeline=test_pipeline,
        classes=['pedestrian']))