from typing import Optional
import torch.nn as nn

from mmcv.cnn.utils.weight_init import INITIALIZERS, update_init_info
from mmcv.runner import (
    _load_checkpoint_with_prefix,
    load_checkpoint,
    load_state_dict)
from mmcv.utils import get_logger, print_log


@INITIALIZERS.register_module(name='KDPretrained')
class KDPretrainedInit:
    """Initialize a knowledge distillation module by loading a pretrained 
    student model and copying its weights into the corresponding teacher
    parameters.

    Args:
        checkpoint (str): the checkpoint file of the pretrained model should
            be load.
        prefix (str, optional): the prefix of a sub-module in the pretrained
            model. it is for loading a part of the pretrained model to
            initialize. For example, if we would like to only load the
            backbone of a detector model, we can set ``prefix='backbone.'``.
            Defaults to None.
        map_location (str): map tensors into proper locations.
    """

    def __init__(self,
                 checkpoint: str,
                 prefix: Optional[str] = None,
                 map_location: Optional[str] = None):
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.map_location = map_location

    def _update_state_dict_with_teacher(self, module, state_dict, logger):
        new_state_dict = state_dict.copy()
        # check if state_dict contains teacher weights
        contains_teacher = any(
            key.startswith('teacher_') for key in state_dict.keys()
        )
        # if not, augment the state_dict with teacher keys and copy values
        if not contains_teacher:
            from mmcv.parallel import is_module_wrapper
            if is_module_wrapper(module):
                module = module.module
            
            keys_mapping = []
            for key in module.state_dict().keys():
                if key.startswith('teacher_'):
                    keys_mapping.append((key[len('teacher_'):], key))
            print_log(
                'initializing the following teacher keys from student weights: '
                f'{[k[1] for k in keys_mapping]}', logger=logger)
            
            for k, v in state_dict.items():
                for pattern, replacement in keys_mapping:
                    k = k.replace(pattern, replacement)
                new_state_dict[k] = v
        return new_state_dict

    def _load_teacher(self, module, checkpoint, logger):
        # Check if state_dict already contains teacher weights
        state_dict = checkpoint["state_dict"]
        state_dict = self._update_state_dict_with_teacher(module, state_dict, logger)
        checkpoint["state_dict"] = state_dict

        # Load updated checkpoint with teacher keys
        load_state_dict(module, state_dict, strict=False, logger=logger)

    def __call__(self, module: nn.Module) -> None:
        logger = get_logger('mmcv')
        if self.prefix is None:
            print_log(f'load model from: {self.checkpoint}', logger=logger)
            checkpoint = load_checkpoint(
                module,
                self.checkpoint,
                map_location=self.map_location,
                strict=False,
                logger=logger)
            self._load_teacher(module, checkpoint, logger=logger)
        else:
            print_log(
                f'load {self.prefix} in model from: {self.checkpoint}',
                logger=logger)
            state_dict = _load_checkpoint_with_prefix(
                self.prefix, self.checkpoint, map_location=self.map_location)
            state_dict = self._update_state_dict_with_teacher(
                module, state_dict, logger)
            load_state_dict(module, state_dict, strict=False, logger=logger)

        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}: load from {self.checkpoint}'
        return info