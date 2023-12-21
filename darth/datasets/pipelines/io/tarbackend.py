import io
import tarfile
import numpy as np
from PIL import Image

from mmcv import BaseStorageBackend, FileClient


@FileClient.register_backend('tar')
class TarBackend(BaseStorageBackend):

    def __init__(self, tar_path="", **kwargs):
        self.tar_path = str(tar_path)
        self.client = tarfile.TarFile(self.tar_path)

    def get(self, filepath):
        """Get values according to the filepath.
        Args:
            filepath (str | obj:`Path`): Here, filepath is the tar key.
        """
        filepath = str(filepath)
        data = self.client.extractfile(filepath)
        data = data.read()
        # data = Image.open(io.BytesIO(data))
        # data = np.array(data)
        return data

    def get_text(self, filepath):
        raise NotImplementedError

    def __del__(self):
        self.client.close()