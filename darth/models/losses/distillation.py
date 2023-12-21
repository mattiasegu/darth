import torch
import torch.nn as nn
from torch.nn.functional import l1_loss, mse_loss

from mmdet.models import LOSSES


@LOSSES.register_module()
class RPNDistillationLoss(nn.Module):
    """RPNDistillationLoss
    Args:
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        reg_valid_threshold (float, optional): Threshold over which applying
            regression distillation. Default to 0.1.
    """

    def __init__(self, loss_weight=1.0, reg_valid_threshold=0.1):
        super(RPNDistillationLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reg_valid_threshold = reg_valid_threshold
    
    def forward(self, inputs, targets, **kwargs):

        """Forward pass.
        Args:
            inputs: Dictionary of classification scores and bounding box
            refinements for the sampled proposals. For cls scores, the shape is
            (b*n) * (cats + 1), where n is sampled proposal in each image, cats
            is the total number of categories without the background. For bbox
            preds, the shape is (b*n) * (4*cats)
            targets: Same output by roi from the teacher output.
        Returns:
            The ROI distillation loss.
        """
        rpn_student_cls = inputs["cls_scores"]
        rpn_teacher_cls = targets["cls_scores"]
        rpn_student_reg = inputs["bbox_preds"]
        rpn_teacher_reg = targets["bbox_preds"]

        cls_loss = []
        anchor_num = []
        for rpn_student_cls_current, rpn_teacher_cls_current in zip(
            rpn_student_cls, rpn_teacher_cls
        ):
            loss_current = mse_loss(
                rpn_teacher_cls_current,
                rpn_student_cls_current,
                reduction="none",
            )
            cls_loss.append(torch.sum(loss_current))  # type: ignore
            anchor_num.append(loss_current.numel())
        cls_result = sum(cls_loss) / sum(anchor_num)

        reg_loss = []
        for rpn_student_reg_current, rpn_teacher_reg_current, ss, tt in zip(
            rpn_student_reg, rpn_teacher_reg, rpn_student_cls, rpn_teacher_cls
        ):
            reg_mask = tt - ss > self.reg_valid_threshold
            reg_mask = reg_mask.long()
            reg_mask = torch.repeat_interleave(  # type: ignore
                reg_mask, 4, dim=1
            )
            loss_current = mse_loss(
                rpn_teacher_reg_current,
                rpn_student_reg_current,
                reduction="none",
            )
            assert reg_mask.shape == loss_current.shape
            loss_current *= reg_mask
            reg_loss.append(torch.sum(loss_current))  # type: ignore
        reg_result = sum(reg_loss) / sum(anchor_num)

        result = cls_result + reg_result

        return result * self.loss_weight  # type: ignore



@LOSSES.register_module()
class ROIDistillationLoss(nn.Module):
    """ROIDistillationLoss
    Args:
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(ROIDistillationLoss, self).__init__()
        self.loss_weight = loss_weight
    
    def forward(self, inputs, targets, **kwargs):

        """Forward pass.
        Args:
            inputs: Dictionary of classification scores and bounding box
            refinements for the sampled proposals. For cls scores, the shape is
            (b*n) * (cats + 1), where n is sampled proposal in each image, cats
            is the total number of categories without the background. For bbox
            preds, the shape is (b*n) * (4*cats)
            targets: Same output by roi from the teacher output.
        Returns:
            The ROI distillation loss.
        """
        roi_teacher_cls = targets["cls_score"]
        roi_teacher_reg = targets["bbox_pred"]
        roi_student_cls = inputs["cls_score"]
        roi_student_reg = inputs["bbox_pred"]

        # subtract the mean of cls score over the cls dimension
        roi_teacher_cls -= torch.mean(  # type: ignore
            roi_teacher_cls, dim=-1, keepdim=True
        )
        roi_student_cls -= torch.mean(  # type: ignore
            roi_student_cls, dim=-1, keepdim=True
        )

        assert roi_student_cls.shape == roi_teacher_cls.shape
        assert roi_student_reg.shape == roi_teacher_reg.shape

        total_num = roi_teacher_cls.numel()
        cls_loss = mse_loss(roi_teacher_cls, roi_student_cls, reduction="none")
        reg_loss = mse_loss(roi_teacher_reg, roi_student_reg, reduction="none")

        result = torch.sum(cls_loss) + torch.sum(reg_loss)  # type: ignore
        result = result / total_num

        return self.loss_weight * result
