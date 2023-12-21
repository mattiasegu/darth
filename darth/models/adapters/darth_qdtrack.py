import torch
import torchvision
from copy import deepcopy

from mmcv.parallel import collate
from mmdet.datasets.pipelines import Compose
from mmtrack.models.builder import MODELS

from darth.models.utils.transforms import (
    detections2bboxes,
    filter_bboxes_by_confidence,
    bboxes_to_tensor,
)

from .kd_qdtrack import KDQDTrack


def to_results(
    img, bboxes, labels, img_metas, bboxes_ignore=None, instance_ids=None
):
    if instance_ids is None:
        instance_ids = [
            torch.arange(1, len(label)+1).to(label.device) for label in labels]
    results = []
    for i, (im, box, label, id) in enumerate(zip(
        img, bboxes, labels, instance_ids)):
        im = im.cpu().detach().numpy().transpose(1, 2, 0)
        result = dict(
            img=im,
            img_shape=tuple(im.shape),
            img_info={},
            gt_bboxes=box.cpu().detach().numpy(),
            gt_bboxes_ignore=(
                bboxes_ignore[i] if bboxes_ignore else None),
            gt_labels=label.cpu().detach().numpy(),
            gt_instance_ids=id,
            img_metas=img_metas[i],
            img_fields=['img'],
            bbox_fields=(
                ['gt_bboxes', 'gt_bboxes_ignore'] if bboxes_ignore
                else ['gt_bboxes']),
        )
        results.append([result])
    return results


def concatenate_results(results, references):
    for result, reference in zip(results, references):
        result.extend(reference)
    return results


def apply_pipeline(results, pipeline):
    outs = []
    for result in results:
        out = pipeline(result)
        outs.append(out)
    return outs


def format_results(results, samples_per_gpu=1, ref_prefix='c'):
    from mmtrack.datasets.pipelines import SeqDefaultFormatBundle
    formatter = SeqDefaultFormatBundle(ref_prefix=ref_prefix)
    outs = []
    for result in results:
        out = formatter(result)
        outs.append(out)
    outs = collate(outs, samples_per_gpu=samples_per_gpu)
    return outs


def parse_results(results, device, keys=[]):
    fields_to_return = []
    for key in keys:
        if key in results:
            data = results[key].data[0]
            if 'img_metas' in key:
                pass
            elif isinstance(data, list):
                data = [d.to(device) for d in data]
            else:
                data = data.to(device)
            fields_to_return.append(data)
        else:
            fields_to_return.append(None)
    return fields_to_return


def drop_empty_images(
    img, bboxes, labels, match_indices, img_metas, bboxes_ignore,
    ref_img, ref_bboxes, ref_labels, ref_match_indices, ref_img_metas, ref_bboxes_ignore,    
):
    """Drop images without bboxes."""
    keep_img = []
    keep_bboxes = []
    keep_labels = []
    keep_match_indices = []
    keep_img_metas = []
    keep_bboxes_ignore = [] if bboxes_ignore is not None else None
    ref_keep_img = []
    ref_keep_bboxes = []
    ref_keep_labels = []
    ref_keep_match_indices = []
    ref_keep_img_metas = []
    ref_keep_bboxes_ignore = [] if ref_bboxes_ignore is not None else None
    for i, bbox in enumerate(bboxes):
        if len(bbox) > 0:
            keep_img.append(img[i])
            keep_bboxes.append(bboxes[i])
            keep_labels.append(labels[i])
            keep_match_indices.append(match_indices[i])
            keep_img_metas.append(img_metas[i])
            if bboxes_ignore is not None:
                keep_bboxes_ignore.append(bboxes_ignore[i])
            ref_keep_img.append(ref_img[i])
            ref_keep_bboxes.append(ref_bboxes[i])
            ref_keep_labels.append(ref_labels[i])
            ref_keep_match_indices.append(ref_match_indices[i])
            ref_keep_img_metas.append(ref_img_metas[i])
            if ref_bboxes_ignore is not None:
                ref_keep_bboxes_ignore.append(ref_bboxes_ignore[i])
    
    keep_img = torch.stack(keep_img) if len(keep_img) > 0 else None
    ref_keep_img = torch.stack(ref_keep_img) if len(ref_keep_img) > 0 else None
    return (
        keep_img, keep_bboxes, keep_labels, keep_match_indices,
        keep_img_metas, keep_bboxes_ignore,
        ref_keep_img, ref_keep_bboxes, ref_keep_labels, ref_keep_match_indices,
        ref_keep_img_metas, ref_keep_bboxes_ignore,
    )


@MODELS.register_module()
class DARTHQDTrack(KDQDTrack):
    """Implementation of `DARTH <https://arxiv.org/abs/2310.01926>`_, a domain adaptive MOT framework based on `QDTrack
    <https://arxiv.org/abs/2006.06664>`_.
    Args:
        transforms list(dict): Configuration of transforms for data augmentation
            of keyframe into reference frame.
    """

    def __init__(self,
                 teacher_pipeline,
                 student_pipeline,
                 contrastive_pipeline,
                 out_pipeline,
                 conf_thr=0.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # build augmentation pipeline
        self.teacher_pipeline = Compose(teacher_pipeline)
        self.student_pipeline = Compose(student_pipeline)
        self.contrastive_pipeline = Compose(contrastive_pipeline)
        self.out_pipeline = Compose(out_pipeline)
        
        # confidence threshold for teacher bboxes selection
        self.conf_thr = conf_thr

    def _prepare_views(self,
                       img,
                       img_metas,
                       gt_bboxes,
                       gt_labels,
                       gt_bboxes_ignore=None,
                       gt_masks=None):
        """Prepare teacher, student and contrastive views for the self-supervised tracking framework."""
        keys = [
            'img', 'gt_bboxes', 'gt_labels', 'img_metas', 'gt_instance_ids',
            'gt_bboxes_ignore']

        # process teacher
        t_results = to_results(
            img,
            gt_bboxes,
            gt_labels,
            img_metas,
            bboxes_ignore=gt_bboxes_ignore,
            instance_ids=None)
        t_t_processed = concatenate_results(deepcopy(t_results), deepcopy(t_results))
        t_t_processed = apply_pipeline(t_t_processed, self.out_pipeline)
        t_t_results = format_results(
            t_t_processed, samples_per_gpu=img.shape[0], ref_prefix='t'
        )
        (
            gt_img, gt_bboxes, gt_labels, img_metas, gt_instance_ids, gt_bboxes_ignore
        ) = parse_results(t_t_results, img.device, keys=keys)

        # teacher forward pass to generate pseudo labels
        t_x = self.teacher_detector.extract_feat(gt_img)
        t_proposal_list = self.teacher_detector.rpn_head.simple_test_rpn(
            t_x, img_metas)
        t_det_results = self.teacher_detector.roi_head.simple_test(
            t_x, t_proposal_list, img_metas, rescale=False)

        t_bboxes, t_labels, t_confidences = detections2bboxes(
            t_det_results)
        t_bboxes, t_labels, t_confidences = filter_bboxes_by_confidence(
            t_bboxes, t_labels, t_confidences, conf_thr=self.conf_thr)
        t_bboxes, t_labels, t_confidences = bboxes_to_tensor(
            t_bboxes, t_labels, t_confidences, img.device)
        
        # augment teacher view for harder distillation
        t_results = to_results(
            img,
            t_bboxes,
            t_labels,
            img_metas,
            bboxes_ignore=gt_bboxes_ignore,
            instance_ids=None)
        t_processed = apply_pipeline(deepcopy(t_results), self.teacher_pipeline)
        t_t_processed = concatenate_results(deepcopy(t_processed), deepcopy(t_processed))
        t_t_processed = apply_pipeline(t_t_processed, self.out_pipeline)
        t_t_results = format_results(
            t_t_processed, samples_per_gpu=img.shape[0], ref_prefix='t'
        )
        (
            t_img, t_bboxes, t_labels, t_img_metas, t_instance_ids, t_bboxes_ignore
        ) = parse_results(t_t_results, img.device, keys=keys)

        # cleaning conflicting fields (if any)
        s_input = deepcopy(t_processed)
        [t[0].pop('scale', 0) for t in s_input]
        [t[0].pop('scale_factor', 0) for t in s_input]
        # augment teacher view to student view
        s_processed = apply_pipeline(s_input, self.student_pipeline)
        # cleaning conflicting fields (if any)
        # start from the view before teacher augmentation to get an independent view
        c_input = deepcopy(t_results)  
        [t[0].pop('scale', 0) for t in c_input]
        [t[0].pop('scale_factor', 0) for t in c_input]
        # augment student view to contrastive view
        c_processed = apply_pipeline(c_input, self.contrastive_pipeline)
        # c_processed = apply_pipeline(s_processed, self.contrastive_pipeline)
        s_c_processed = concatenate_results(deepcopy(s_processed), c_processed)
        s_c_processed = apply_pipeline(s_c_processed, self.out_pipeline)
        s_c_results = format_results(
            s_c_processed, samples_per_gpu=img.shape[0], ref_prefix='c'
        )
        (
            s_img, s_bboxes, s_labels, s_img_metas, s_instance_ids, s_bboxes_ignore
        ) = parse_results(s_c_results, img.device, keys=keys)
        (
            c_img, c_bboxes, c_labels, c_img_metas, c_instance_ids, c_bboxes_ignore
        ) = parse_results(s_c_results, img.device, keys=['c_' + key for key in keys])
        s_match_indices, c_match_indices = parse_results(
            s_c_results, img.device, keys=['gt_match_indices', 'c_gt_match_indices'])

        # pad img to same size as s_img for correct distillation
        import torch.nn.functional as F
        t_img = F.pad(
            t_img, pad=(
                0, s_img.shape[3] - t_img.shape[3],
                0, s_img.shape[2] - t_img.shape[2],
            ), mode='constant', value=0
        )

        # visualization (useful when debugging)
        if False:
            import matplotlib.pyplot as plt
            from ..utils import get_det_im
            imgs = []
            columns = 4
            # rows = 1
            rows = img.shape[0]
            for i in range(0, rows):
                imgs.append(get_det_im(gt_img, gt_bboxes, gt_labels, i, img_metas[0]['img_norm_cfg']))
                imgs.append(get_det_im(t_img, t_bboxes, t_labels, i, t_img_metas[0]['img_norm_cfg']))
                imgs.append(get_det_im(s_img, s_bboxes, s_labels, i, s_img_metas[0]['img_norm_cfg']))
                imgs.append(get_det_im(c_img, c_bboxes, c_labels, i, c_img_metas[0]['img_norm_cfg']))
            fig = plt.figure(figsize=(8, 8))
            for i in range(0, rows*columns):
                fig.add_subplot(rows, columns, i+1)
                plt.imshow(imgs[i])
                if False:
                    plt.imsave(f'tmp/img_{i}.jpg', imgs[i])
                plt.axis('off')
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()

        t_view = t_img, t_bboxes, t_labels, None, t_img_metas, t_bboxes_ignore
        s_view = s_img, s_bboxes, s_labels, s_match_indices, s_img_metas, s_bboxes_ignore
        c_view = c_img, c_bboxes, c_labels, c_match_indices, c_img_metas, c_bboxes_ignore
        return t_view, s_view, c_view

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
            dict[str : TensorNone]: All losses.
        """
        # basic assertions
        assert self.detector.with_rpn, "two-stage KDQDT must have rpn"
        assert self.with_teacher_detector, "teacher_detector must exist"
        assert gt_masks is None, "gt_masks is not supported"

        with torch.no_grad():
            t_view, s_view, c_view = self._prepare_views(
                img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks)

        img, gt_bboxes, gt_labels, _, img_metas, gt_bboxes_ignore = t_view
        s_img, s_bboxes, s_labels, s_match_indices, s_img_metas, s_bboxes_ignore = s_view
        c_img, c_bboxes, c_labels, c_match_indices, c_img_metas, c_bboxes_ignore = c_view
        
        # student training
        losses = dict()

        # teacher forward pass on padded teacher view
        with torch.no_grad():
            t_x = self.teacher_detector.extract_feat(img)
            t_proposal_list = self.teacher_detector.rpn_head.simple_test_rpn(
                t_x, img_metas)

        # key frame forward
        s_x = self.detector.extract_feat(s_img)
        s_proposal_list = self.detector.rpn_head.simple_test_rpn(s_x, s_img_metas)

        # rpn distillation loss
        if (
            self.rpn_distillation_loss is not None and
            self.rpn_distillation_loss.loss_weight > 0
        ):
            rpn_outs = self.get_rpn_displacement(self.detector, s_x)
            with torch.no_grad():
                teacher_rpn_outs = self.get_rpn_displacement(
                    self.teacher_detector, t_x)
            rpn_distillation_loss = self.rpn_distillation_loss(
                rpn_outs, teacher_rpn_outs)
            losses.update({"rpn_distillation_loss": rpn_distillation_loss})

        # roi distillation loss
        if (
            self.roi_distillation_loss is not None and
            self.roi_distillation_loss.loss_weight > 0
        ):
            roi_outs = self.get_roi_displacement(
                self.detector, s_x, t_proposal_list)
            with torch.no_grad():
                teacher_roi_outs = self.get_roi_displacement(
                    self.teacher_detector, t_x, t_proposal_list)
            roi_distillation_loss = self.roi_distillation_loss(
                roi_outs, teacher_roi_outs)
            losses.update({"roi_distillation_loss": roi_distillation_loss})

        (
            s_img, s_bboxes, s_labels, s_match_indices, s_img_metas, s_bboxes_ignore,
            c_img, c_bboxes, c_labels, c_match_indices, c_img_metas, c_bboxes_ignore,
        ) = drop_empty_images(
            s_img, s_bboxes, s_labels, s_match_indices, s_img_metas, s_bboxes_ignore,
            c_img, c_bboxes, c_labels, c_match_indices, c_img_metas, c_bboxes_ignore,
        )
        
        if (
            s_img is not None and (
                self.track_head.embed_head.loss_track.loss_weight > 0 or
                self.track_head.embed_head.loss_track_aux.loss_weight > 0)
        ):
            # ref frame forward
            c_x = self.detector.extract_feat(c_img)
            c_proposal_list = self.detector.rpn_head.simple_test_rpn(
                c_x, c_img_metas)
            track_losses = self.track_head.forward_train(
                s_x, s_img_metas, s_proposal_list, s_bboxes, s_labels,
                s_match_indices, c_x, c_img_metas, c_proposal_list,
                c_bboxes, c_labels, s_bboxes_ignore, gt_masks,
                c_bboxes_ignore)
        else:
            track_losses = {
                'loss_track': torch.tensor(0.0).to(img.device),
                'loss_track_aux': torch.tensor(0.0).to(img.device)}
        losses.update(track_losses)

        # print(losses)
        return losses
