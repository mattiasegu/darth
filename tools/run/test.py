# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time

import mmcv
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import set_random_seed
from mmtrack.core import setup_multi_processes
from mmtrack.utils import get_root_logger

from darth.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='mmtrack test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--results-path',
        type=str,
        help='path where to read results from')
    parser.add_argument(
        '--work-dir',
        default='./work_dirs',
        help='the dir to save logs and models'
    )
    parser.add_argument('--exp-name', help='experiment folder name')
    parser.add_argument('--version', help='version folder name')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument('--eval', type=str, nargs='+', help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--not_save_results',
        action='store_true',
        help='whether to skip saving results and metrics to work_dir.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.eval or args.format_only or args.show or args.show_dir, (
        'Please specify at least one operation (save/eval/format/show the '
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if (
        args.results_path is not None and
        not args.results_path.endswith(('.pkl', '.pickle'))
    ):
        raise ValueError('The results file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if cfg.get('USE_MMDET', False):
        from mmdet.apis import multi_gpu_test, single_gpu_test
        from mmdet.datasets import build_dataloader
        from mmdet.models import build_detector as build_model
        if 'detector' in cfg.model:
            cfg.model = cfg.model.detector
    elif cfg.get('TRAIN_REID', False):
        from mmdet.apis import multi_gpu_test, single_gpu_test
        from mmdet.datasets import build_dataloader

        from mmtrack.models import build_reid as build_model
        if 'reid' in cfg.model:
            cfg.model = cfg.model.reid
    else:
        from mmtrack.apis import multi_gpu_test, single_gpu_test
        from mmtrack.datasets import build_dataloader
        from darth.models import build_model
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set random seeds. Force setting fixed seed and deterministic=True in SOT
    # configs
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('seed', None) is not None:
        set_random_seed(
            cfg.seed, deterministic=cfg.get('deterministic', False))
    cfg.data.test.test_mode = True

    cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # work_dir is determined in this priority: CLI > segment in file > filename
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = args.exp_name if args.exp_name is not None else osp.splitext(
        osp.basename(args.config))[0]
    version = args.version if args.version is not None else timestamp
    cfg.work_dir = osp.join(args.work_dir, exp_name, version)

    rank, _ = get_dist_info()
    # allows not to create
    if cfg.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # init the logger before other steps
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
        json_file = osp.join(cfg.work_dir, f'eval_{timestamp}.log.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    if args.results_path is not None:
        print(f'Reading results from {args.results_path}')
        outputs = mmcv.load(args.results_path)
    else:
        # build the model and load checkpoint
        if cfg.get('test_cfg', False):
            model = build_model(
                cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        else:
            model = build_model(cfg.model)
        # We need call `init_weights()` to load pretained weights in MOT task.
        model.init_weights()
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if args.checkpoint is not None:
            checkpoint = load_checkpoint(
                model, args.checkpoint, map_location='cpu')
            if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
        if not hasattr(model, 'CLASSES'):
            model.CLASSES = dataset.CLASSES

        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)

        if not distributed:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            outputs = single_gpu_test(
                model,
                data_loader,
                args.show,
                args.show_dir,
                show_score_thr=args.show_score_thr)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                    args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.results_path is None:
            results_path = osp.join(cfg.work_dir, "raw_results.pkl")
            print(f'Writing results to {results_path}')
            mmcv.dump(outputs, results_path)
        kwargs = {} if args.eval_options is None else args.eval_options
        if not args.not_save_results:
            kwargs.update(dict(results_path=cfg.work_dir))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            eval_hook_args = [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'by_epoch'
            ]
            for key in eval_hook_args:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, logger=logger, **eval_kwargs)
            metric_dict = dict(
                config=args.config, mode='test', epoch=cfg.total_epochs)
            metric_dict.update(metric)
            if cfg.work_dir is not None:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()
