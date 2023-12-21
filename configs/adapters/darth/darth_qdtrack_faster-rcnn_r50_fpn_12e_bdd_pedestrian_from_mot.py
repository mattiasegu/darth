# this config file can be used for both mot and dancetrack models
_base_ = [
    '../../_base_/target_datasets/bdd_pedestrian_from_mot.py',
    './darth_qdtrack_faster_rcnn_r50_fpn_4e_base.py',
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
    out_pipeline=out_pipeline
)