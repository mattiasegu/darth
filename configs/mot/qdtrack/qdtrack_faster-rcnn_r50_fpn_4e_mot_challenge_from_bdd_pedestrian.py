_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_4e_base.py',
    '../../_base_/datasets/mot_challenge_from_bdd.py'
]
model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=1))),
)