from copy import deepcopy
from kornia import augmentation as kornia_augmentation


def build_transform(cfg):
    assert hasattr(cfg, "type")
    kornia_cfg = deepcopy(cfg)
    kornia_type = kornia_cfg.pop("type")
    augmentation = getattr(kornia_augmentation, kornia_type)
    return augmentation(**kornia_cfg)