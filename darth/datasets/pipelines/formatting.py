import torch

from mmdet.datasets.builder import PIPELINES



def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()


@PIPELINES.register_module()
class TensorToNumpy:
    """Convert some results from :obj:`torch.Tensor` to :obj:`np.ndarray` by
    given keys.
    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """
        for key in self.keys:
            results[key] = to_numpy(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'