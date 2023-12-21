_base_ = [
    '../../_base_/target_datasets/mot_challenge_from_bdd.py',
    './darth_qdtrack_faster_rcnn_r50_fpn_4e_base.py',
]

norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
    out_pipeline=out_pipeline
)
