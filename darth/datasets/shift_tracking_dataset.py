from mmdet.datasets import DATASETS

from .coco_tracking_dataset import CocoTrackingDataset


@DATASETS.register_module()
class SHIFTTrackingDataset(CocoTrackingDataset):
    """Dataset for SHIFT: https://www.vis.xyz/shift/."""

    CLASSES = (
        "pedestrian",
        "car",
        "bus",
        "truck",
        "bicycle",
        "motorcycle",
    )

