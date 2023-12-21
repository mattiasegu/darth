# dataset settings
dataset_type = 'BDDTrackingDataset'
data_root = 'data/bdd100k/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    # comment above line and comment out the lines below if use hdf5 file.
    # dict(type='LoadMultiImagesFromFile',
    #      file_client_args=dict(
    #          img_db_path= 'data/bdd/hdf5s/100k_train.hdf5',
    #          # vid_db_path='data/bdd/hdf5s/track_train.hdf5',
    #          backend='hdf5',
    #          type='bdd')),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        img_scale=[(1296, 640), (1296, 672), (1296, 704), (1296, 736),
                   (1296, 768), (1296, 800), (1296, 720)],
        share_params=False,
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=False, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices'],
        ),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # comment above line and comment out the lines below if use hdf5 file.
    # dict(type='LoadImageFromFile',
    #      file_client_args=dict(
    #          vid_db_path='data/bdd/hdf5s/track_val.hdf5',
    #          backend='hdf5',
    #          type='bdd')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1296, 720),
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
            'annotations/box_track_20/box_track_train_cocofmt.json',
            img_prefix=data_root + 'images/track/train/',
            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(num_ref_imgs=1, frame_range=3, method='uniform'),
            pipeline=train_pipeline,
            classes=['pedestrian']),
        dict(
            type=dataset_type,
            load_as_video=False,
            ann_file=data_root + 'annotations/det_20/box_det_train_cocofmt.json',
            img_prefix=data_root + 'images/100k/train/',
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