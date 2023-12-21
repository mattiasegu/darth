_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_12e_base.py',
    '../../_base_/datasets/shift_track.py'
]
model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))),
)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[4, 5])
total_epochs = 6
