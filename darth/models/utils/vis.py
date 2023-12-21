import matplotlib.pyplot as plt
import numpy as np
import mmcv
from mmdet.core.visualization import imshow_det_bboxes


def get_det_im(
    img, bboxes, labels, idx,
    img_norm_cfg = dict(
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False)
):
    im = mmcv.imdenormalize(
        img[idx].permute(1,2,0).cpu().numpy(),
        np.array(img_norm_cfg['mean']),
        np.array(img_norm_cfg['std']),
        not img_norm_cfg['to_rgb'])
    det_im = imshow_det_bboxes(
        im, bboxes=bboxes[idx].cpu().numpy(),
        labels=labels[idx].cpu().numpy(), show=False)
    return det_im/255.
