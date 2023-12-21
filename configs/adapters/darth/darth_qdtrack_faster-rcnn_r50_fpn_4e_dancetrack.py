_base_ = [
    '../../_base_/target_datasets/dancetrack.py',
    './darth_qdtrack_faster_rcnn_r50_fpn_4e_base.py',
]

model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=1))),
)