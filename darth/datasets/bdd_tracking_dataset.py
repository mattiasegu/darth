from mmdet.datasets import DATASETS

from .coco_tracking_dataset import CocoTrackingDataset


@DATASETS.register_module()
class BDDTrackingDataset(CocoTrackingDataset):
    """Dataset for BDD100K: https://www.bdd100k.com/."""

    CLASSES = (
        "pedestrian",
        "rider",
        "car",
        "bus",
        "truck",
        "bicycle",
        "motorcycle",
        "train",
    )

