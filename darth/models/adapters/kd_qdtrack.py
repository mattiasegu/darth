from mmdet.models import build_loss

from mmtrack.models.builder import MODELS

from .base import TeacherQDTrack


@MODELS.register_module()
class KDQDTrack(TeacherQDTrack):
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
                 loss_rpn_distillation=dict(
                     type='RPNDistillationLoss',
                     loss_weight=1.0,
                     reg_valid_threshold=0.1),
                 loss_roi_distillation=dict(
                     type='ROIDistillationLoss', loss_weight=1.0),
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # build distillation losses
        self.rpn_distillation_loss = build_loss(loss_rpn_distillation)
        self.roi_distillation_loss = build_loss(loss_roi_distillation)
        
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        """Forward function during training.
         Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                each item has a shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of all images.
                each has a shape (num_gts,).
            gt_match_indices (list(Tensor)): Mapping from gt_instance_ids to
                ref_gt_instance_ids of the same tracklet in a pair of images.
            ref_img (Tensor): of shape (N, C, H, W) encoding input reference
                images. Typically these should be mean centered and std scaled.
            ref_img_metas (list[dict]): list of reference image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape',
                and 'img_norm_cfg'.
            ref_gt_bboxes (list[Tensor]): Ground truth bboxes of the
                reference image, each item has a shape (num_gts, 4).
            ref_gt_labels (list[Tensor]): Ground truth labels of all
                reference images, each has a shape (num_gts,).
            gt_masks (list[Tensor]) : Masks for each bbox, has a shape
                (num_gts, h , w).
            gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes to be
                ignored, each item has a shape (num_ignored_gts, 4).
            ref_gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes
                of reference images to be ignored,
                each item has a shape (num_ignored_gts, 4).
            ref_gt_masks (list[Tensor]) : Masks for each reference bbox,
                has a shape (num_gts, h , w).
        Returns:
            dict[str : Tensor]: All losses.
        """
        # basic assertions
        assert self.detector.with_rpn, "two-stage KDQDT must have rpn"
        assert self.with_teacher_detector, "teacher_detector must exist"
        import torch

        # visualization (useful when debugging)
        # from ..utils import get_det_im
        # import matplotlib.pyplot as plt
        # imgs = []
        # columns = 2
        # rows = img.shape[0]
        # for i in range(0, rows):
        #     imgs.append(get_det_im(img, gt_bboxes, gt_labels, i))
        #     imgs.append(get_det_im(ref_img, ref_gt_bboxes, ref_gt_labels, i))
        # fig = plt.figure(figsize=(8, 8))
        # for i in range(0, rows*columns):
        #     fig.add_subplot(rows, columns, i+1)
        #     plt.imshow(imgs[i])
        #     plt.axis('off')
        # plt.show()

        # teacher forward pass
        teacher_x = self.teacher_detector.extract_feat(img)
        teacher_proposal_list = self.teacher_detector.rpn_head.simple_test_rpn(
            teacher_x, img_metas)
        teacher_det_results = self.teacher_detector.roi_head.simple_test(
            teacher_x, teacher_proposal_list, img_metas, rescale=False)

        # student training
        losses = dict()

        # key frame forward
        x = self.detector.extract_feat(img)
        proposal_list = self.detector.rpn_head.simple_test_rpn(x, img_metas)

        # ref frame forward
        ref_x = self.detector.extract_feat(ref_img)
        ref_proposal_list = self.detector.rpn_head.simple_test_rpn(
            ref_x, ref_img_metas)

        # rpn distillation loss
        if (
            self.rpn_distillation_loss is not None and
            self.rpn_distillation_loss.loss_weight > 0
        ):
            rpn_outs = self.get_rpn_displacement(self.detector, x)
            teacher_rpn_outs = self.get_rpn_displacement(
                self.teacher_detector, teacher_x)
            rpn_distillation_loss = self.rpn_distillation_loss(
                rpn_outs, teacher_rpn_outs)
            losses.update({"rpn_distillation_loss": rpn_distillation_loss})

        # roi distillation loss
        if (
            self.roi_distillation_loss is not None and
            self.roi_distillation_loss.loss_weight > 0
        ):
            roi_outs = self.get_roi_displacement(
                self.detector, x, teacher_proposal_list)
            teacher_roi_outs = self.get_roi_displacement(
                self.teacher_detector, teacher_x, teacher_proposal_list)
            roi_distillation_loss = self.roi_distillation_loss(
                roi_outs, teacher_roi_outs)
            losses.update({"roi_distillation_loss": roi_distillation_loss})

        track_losses = self.track_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_match_indices, ref_x, ref_img_metas, ref_proposal_list,
            ref_gt_bboxes, ref_gt_labels, gt_bboxes_ignore, gt_masks,
            ref_gt_bboxes_ignore)

        losses.update(track_losses)

        return losses
