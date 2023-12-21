_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_4e_base.py',
]
model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=1))),
    track_head=dict(train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='CombinedSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(
                    type='IoUBalancedNegSampler',
                    floor_thr=-1,
                    floor_fraction=0,
                    num_bins=3)))),
    tracker=dict(
        type='QuasiDenseTracker',
        init_score_thr=0.7,
        obj_score_thr=0.3,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'))


# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 11])
# checkpoint savingp
checkpoint_config = dict(interval=1)
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(metric=['bbox', 'track'],
    interval=1,
    results_kwargs=dict(
        results_path=None,  # set to None if results saved to tmpdir, to dir to indicate where to save, to .json to load a json
        scalabel=False,  # set to True if you want to submit to the BDD eval
        append_empty_preds=True,  # set to True if you want to submit to the BDD eval
    ),  
    track_kwargs=dict(
        iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_by_classes=False,
        nproc=1,
        majority_voting=False
    ),
)