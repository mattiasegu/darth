import torch

from mmdet.core import bbox2roi
from mmdet.models import build_detector, build_head, build_loss

from mmtrack.core import outs2results, results2outs
from mmtrack.models.builder import MODELS
from mmtrack.models.mot import QDTrack


@MODELS.register_module()
class TeacherQDTrack(QDTrack):
    """Knowledge Distillation module for `QDTrack
    <https://arxiv.org/abs/2006.06664>`_.
    Args:
        detector (dict): Configuration of detector. Defaults to None.
        track_head (dict): Configuration of track head. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
        teacher (dict): Configuration of the teacher model, with args:
            eval_teacher_preds (bool): If True, evaluate the teacher model
                predictions instead of the student ones. Defaults to False.
            eval_mode_teacher (bool): If True, set the teacher in eval mode, 
                i.e. all layers (e.g. dropout, batchnorm, ...) will function in
                eval mode. Defaults to True.
    """

    def __init__(self,
                 detector=None,
                 track_head=None,
                 tracker=None,
                 teacher=dict(
                    eval_teacher_preds=False,
                    eval_mode_teacher=True,),
                 *args,
                 **kwargs):
        super().__init__(detector, track_head, tracker, *args, **kwargs)

        self.eval_teacher_preds = teacher["eval_teacher_preds"]
        self.eval_mode_teacher = teacher["eval_mode_teacher"]

        # build teacher modules
        if detector is not None:
            self.teacher_detector = build_detector(detector)
        if track_head is not None:
            self.teacher_track_head = build_head(track_head)

        # freeze teacher
        self.freeze_module('teacher_detector')
        self.freeze_module('teacher_track_head')
    

    def init_weights(self) -> None:
        """Initialize the weights."""
        import warnings
        from collections import defaultdict
        from mmcv.utils.logging import logger_initialized, print_log

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, '_params_init_info'):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info: defaultdict = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param][
                    'init_info'] = f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                self._params_init_info[param][
                    'tmp_mean_value'] = param.data.mean()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else 'mmcv'

        from mmcv.cnn import initialize
        from mmcv.cnn.utils.weight_init import update_init_info
        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {module_name} with init_cfg {self.init_cfg}',
                    logger=logger_name)
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of
                    # the pre-trained model
                    # from being overwritten by
                    # the `init_weights`
                    if 'Pretrained' in self.init_cfg['type']:
                        return

            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f'Initialized by '
                        f'user-defined `init_weights`'
                        f' in {m.__class__.__name__} ')

            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

        if is_top_level_module:
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                del sub_module._params_init_info


    @property
    def with_teacher_detector(self):
        """bool: whether the framework has a detector."""
        return hasattr(
            self, 'teacher_detector'
        ) and self.teacher_detector is not None

    @property
    def with_teacher_reid(self):
        """bool: whether the framework has a reid model."""
        return hasattr(
            self, 'teacher_reid'
        ) and self.teacher_reid is not None

    @property
    def with_teacher_motion(self):
        """bool: whether the framework has a motion model."""
        return hasattr(
            self, 'teacher_motion'
        ) and self.teacher_motion is not None

    @property
    def with_teacher_track_head(self):
        """bool: whether the teacher model has a track_head."""
        return hasattr(
            self, 'teacher_track_head'
        ) and self.teacher_track_head is not None
    
    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_mode_teacher:
            self.teacher_detector.train(False)
            self.teacher_track_head.train(False)
        else:
            self.teacher_detector.train(mode)
            self.teacher_track_head.train(mode)
        super().train(mode)

    def get_rpn_displacement(self, detector, x):
        """Get rpn raw output for cls and reg given backbone features.
        Args:
            detector: a detector model.
            x: feature pyramid.
        Returns:
            rpn displacement dict, containing keys cls_scores and bbox_preds for
            each pyramid level. 
            bbox_preds has shape (batch_size, num_anchors*4, height, width)
        """
        assert x is not None, "dense_head requires features"

        cls_scores, bbox_preds = detector.rpn_head.forward(x)
        rpn_output = dict(cls_scores=cls_scores, bbox_preds=bbox_preds)
        return rpn_output

    def get_roi_displacement(self, detector, x, proposal_list):
        """Get roi raw output for cls and reg given proposals.
        Args:
            detector: a detector model.
            x: feature pyramid.
            proposal_list: proposals from RPN.
        Returns:
            roi output dict, containing keys cls_score and bbox_pred.
        """
        assert x is not None, "roi_head requires features"

        rois = bbox2roi(proposal_list)
        roi_output = detector.roi_head._bbox_forward(x, rois)
        # NB: roi_output["cls_score"] is not after softmax

        return roi_output

    def simple_test(self, img, img_metas, rescale=False):
        """Test forward.

         Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): whether to rescale the bboxes.

        Returns:
            dict[str : Tensor]: Track results.
        """
        assert self.with_track_head, 'track head must be implemented.'  # noqa
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        if self.eval_teacher_preds:
            detector = self.teacher_detector
            track_head = self.teacher_track_head
        else:
            detector = self.detector
            track_head = self.track_head

        x = detector.extract_feat(img)
        proposal_list = detector.rpn_head.simple_test_rpn(x, img_metas)
        det_results = detector.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

        bbox_results = det_results[0]
        num_classes = len(bbox_results)
        outs_det = results2outs(bbox_results=bbox_results)

        det_bboxes = torch.tensor(outs_det['bboxes']).to(img)
        det_labels = torch.tensor(outs_det['labels']).to(img).long()

        track_bboxes, track_labels, track_ids = self.tracker.track(
            img_metas=img_metas,
            feats=x,
            track_head=track_head,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id)

        track_bboxes = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)['bbox_results']

        return dict(det_bboxes=bbox_results, track_bboxes=track_bboxes)