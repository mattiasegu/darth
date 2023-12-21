_base_ = [
    '../../mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_base.py',
]

# output of student_pipeline is passed to student
teacher_pipeline = [
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqResize',
        img_scale=(1088, 1088),
        share_params=True,
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
        bbox_clip_border=False),
    dict(
        type='SeqRandomCrop',
        share_params=False,
        crop_size=(1088, 1088),
        bbox_clip_border=False,
        allow_negative_crop=True,
        ),
]

# output of teacher_pipeline + student_pipeline is passed to student
# NB: no geometric augmentations are allowed here
# (teacher and student geometric view must be same to allow distillation)
student_pipeline = [
    dict(type='SeqPhotoMetricDistortion', share_params=True),
]

# output of contrastive_pipeline is used as contrastive view
contrastive_pipeline = [
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqResize',
        img_scale=(1088, 1088),
        share_params=True,
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
        bbox_clip_border=False),
    dict(
        type='SeqRandomCrop',
        share_params=False,
        crop_size=(1088, 1088),
        bbox_clip_border=False,
        allow_negative_crop=True,
        ),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
]

norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
out_pipeline = [
    dict(type='SeqNormalize', **norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=False),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
            'gt_instance_ids'
        ]),
]

model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=1))),
    type='DARTHQDTrack',
    init_cfg=dict(
        type='KDPretrained',
        checkpoint='https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635-76f295ef.pth',
        map_location='cpu',
    ),
    tracker=dict(
        type='QDTracker',
    ),
    teacher=dict(
        eval_teacher_preds=False,
        eval_mode_teacher=True,
    ),
    loss_rpn_distillation=dict(
        type='RPNDistillationLoss',
        loss_weight=0.1,
        reg_valid_threshold=0.1
    ),
    loss_roi_distillation=dict(
        type='ROIDistillationLoss', loss_weight=0.1
    ),
    teacher_pipeline=teacher_pipeline,
    student_pipeline=student_pipeline,
    contrastive_pipeline=contrastive_pipeline,
    out_pipeline=out_pipeline,
    conf_thr=0.7,
)

custom_hooks=[
    dict(
        type='EMATrainHook', momentum=0.002, interval=1, warm_up=100
    ),
]

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35.0, norm_type=2))
lr_config = dict(policy='step', step=[3])