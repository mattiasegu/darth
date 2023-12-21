_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_12e_base.py',
    '../../_base_/datasets/bdd_pedestrian_track_det.py'
]
model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=1))),
)