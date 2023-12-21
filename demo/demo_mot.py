import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

from math import ceil
import numpy as np
import cv2
import mmcv
from mmtrack.core import imshow_tracks, results2outs
from mmtrack.core.utils.visualization import random_color

from darth.apis import inference_mot, init_model


def cv2_show_tracks(img,
                     bboxes,
                     labels,
                     ids,
                     masks=None,
                     classes=None,
                     score_thr=0.0,
                     thickness=2,
                     font_scale=0.4,
                     show=False,
                     wait_time=0,
                     out_file=None):
    """Show the tracks with opencv."""
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert ids.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 5
    if isinstance(img, str):
        img = mmcv.imread(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    inds = np.where(bboxes[:, -1] > score_thr)[0]
    bboxes = bboxes[inds]
    labels = labels[inds]
    ids = ids[inds]
    if masks is not None:
        assert masks.ndim == 3
        masks = masks[inds]
        assert masks.shape[0] == bboxes.shape[0]

    text_width, text_height = 9, 13
    text_width = ceil(text_width*font_scale/0.5)
    text_height = ceil(text_height*font_scale/0.5)
    for i, (bbox, label, id) in enumerate(zip(bboxes, labels, ids)):
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = float(bbox[-1])

        # bbox
        bbox_color = random_color(id)
        bbox_color = [int(255 * _c) for _c in bbox_color][::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        # score
        text = '{:.02f}'.format(score)
        if classes is not None:
            text += f'|{classes[label]}'
        width = len(text) * text_width
        img[y1:y1 + text_height, x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            text, (x1, y1 + text_height - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

        # id
        text = str(id)
        img[y1 + text_height:y1 + 2 * text_height,
            x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            str(id), (x1, y1 + 2 * text_height - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

        # mask
        if masks is not None:
            mask = masks[i].astype(bool)
            mask_color = np.array(bbox_color, dtype=np.uint8).reshape(1, -1)
            img[mask] = img[mask] * 0.5 + mask_color * 0.5

    if show:
        mmcv.imshow(img, wait_time=wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img



def show_result(img,
                result,
                classes,
                score_thr=0.0,
                thickness=1,
                font_scale=0.5,
                show=False,
                out_file=None,
                wait_time=0,
                **kwargs):
    """Visualize tracking results.

    Args:
        img (str | ndarray): Filename of loaded image.
        result (dict): Tracking result.
            - The value of key 'track_bboxes' is list with length
            num_classes, and each element in list is ndarray with
            shape(n, 6) in [id, tl_x, tl_y, br_x, br_y, score] format.
            - The value of key 'det_bboxes' is list with length
            num_classes, and each element in list is ndarray with
            shape(n, 5) in [tl_x, tl_y, br_x, br_y, score] format.
        thickness (int, optional): Thickness of lines. Defaults to 1.
        font_scale (float, optional): Font scales of texts. Defaults
            to 0.5.
        show (bool, optional): Whether show the visualizations on the
            fly. Defaults to False.
        out_file (str | None, optional): Output filename. Defaults to None.
        backend (str, optional): Backend to draw the bounding boxes,
            options are `cv2` and `plt`. Defaults to 'cv2'.

    Returns:
        ndarray: Visualized image.
    """
    assert isinstance(result, dict)
    track_bboxes = result.get('track_bboxes', None)
    track_masks = result.get('track_masks', None)
    if isinstance(img, str):
        img = mmcv.imread(img)
    outs_track = results2outs(
        bbox_results=track_bboxes,
        mask_results=track_masks,
        mask_shape=img.shape[:2])
    img = cv2_show_tracks(
        img,
        outs_track.get('bboxes', None),
        outs_track.get('labels', None),
        outs_track.get('ids', None),
        outs_track.get('masks', None),
        classes=classes,
        score_thr=score_thr,
        thickness=thickness,
        font_scale=font_scale,
        show=show,
        out_file=out_file,
        wait_time=wait_time)
    return img


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
    parser.add_argument(
        '--video_output', 
        default=None, 
        help='output video file (mp4 format) or None')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument(
        '--keep_frames',
        action='store_true',
        help='whether to keep the frames when output is a video')
    parser.add_argument(
        '--no_class',
        action='store_true',
        help='whether to not show bbox class')
    parser.add_argument(
        '--thickness', default=4, type=int, help='bbox thickness')
    parser.add_argument(
        '--font_scale', default=0.5, type=float, help='font scale')
    parser.add_argument(
        '--backend',
        choices=['cv2', 'plt'],
        default='cv2',
        help='the backend to visualize the results')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    assert args.output or args.show
    # load images
    if osp.isdir(args.input):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(args.input)),
            key=lambda x: x.split('.')[0])
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(args.input)
        IN_VIDEO = True
    # define output
    if args.output is not None:
        if args.video_output is not None and args.video_output.endswith('.mp4'):
            OUT_VIDEO = True
            if args.keep_frames:
                out_path = osp.dirname(args.output)
                os.makedirs(out_path, exist_ok=True)
            else:                
                out_dir = tempfile.TemporaryDirectory()
                out_path = out_dir.name
            _out = args.video_output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            OUT_VIDEO = False
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)

    fps = args.fps
    if args.show or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    prog_bar = mmcv.ProgressBar(len(imgs))
    # test and show/save the images
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img = osp.join(args.input, img)
        result = inference_mot(model, img, frame_id=i)
        if args.output is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
        else:
            out_file = None
        show_result(
            img,
            result,
            model.CLASSES if not args.no_class else None,
            score_thr=args.score_thr,
            thickness=args.thickness,
            font_scale=args.font_scale,
            show=args.show,
            wait_time=int(1000. / int(fps)) if fps else 0,
            out_file=out_file)
        prog_bar.update()

    if args.output and OUT_VIDEO:
        print(f'making the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.video_output, fps=fps, fourcc='mp4v')
        if not args.keep_frames:
            out_dir.cleanup()


if __name__ == '__main__':
    main()