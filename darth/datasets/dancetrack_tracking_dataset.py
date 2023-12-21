from mmdet.datasets import DATASETS

from .mot_tracking_dataset import MOTTrackingDataset


@DATASETS.register_module()
class DanceTrackTrackingDataset(MOTTrackingDataset):
    """Dataset for DanceTrack: https://github.com/DanceTrack/DanceTrack.
    """

    def get_benchmark_and_eval_split(self):
        """Get benchmark and dataset split to evaluate.

        Get benchmark from upeper/lower-case image prefix and the dataset
        split to evaluate.

        Returns:
            tuple(string): The first string denotes the type of dataset.
            The second string denots the split of the dataset to eval.
        """
        # As DanceTrack only has train/val and use 'val' for evaluation as
        # default, we can directly output the desired split.
        return 'DanceTrack', 'val'
