# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import yaml

import mmcv
import torch
import torch.distributed as dist
torch.multiprocessing.set_sharing_strategy('file_system')

from mmcv import Config, DictAction
from mmcv.runner import init_dist, load_checkpoint
from mmdet.apis import set_random_seed

from mmtrack import __version__
from mmtrack.apis import init_random_seed
from mmtrack.core import setup_multi_processes
from mmtrack.utils import collect_env, get_root_logger

from darth.datasets import build_dataset

# import faulthandler; faulthandler.enable()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work-dir',
        default='./work_dirs',
        help='the dir to save logs and models'
    )
    parser.add_argument('--exp-name', help='experiment folder name')
    parser.add_argument('--version', help='version folder name')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--load-from', help='checkpoint file from which '
        'initializing model weights. The difference from resume-from is that '
        'resume-from resumes training from where it was interrupted, while '
        'checkpoint starts from scratch.')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--no-version',
        action='store_true',
        help='whether to skip version/timestamp in directory naming')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if cfg.get('USE_MMDET', False):
        from mmdet.apis import train_detector as train_model
        from mmdet.models import build_detector as build_model
        if 'detector' in cfg.model:
            cfg.model = cfg.model.detector
    elif cfg.get('TRAIN_REID', False):
        from mmdet.apis import train_detector as train_model

        from mmtrack.models import build_reid as build_model
        if 'reid' in cfg.model:
            cfg.model = cfg.model.reid
    else:
        from mmtrack.apis import train_model
        from darth.models import build_model
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = args.exp_name if args.exp_name is not None else osp.splitext(
        osp.basename(args.config))[0]
    version = args.version if args.version is not None else timestamp
    cfg.work_dir = osp.join(args.work_dir, exp_name)
    if not args.no_version:
        cfg.work_dir = osp.join(cfg.work_dir, version)
    
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds. Force setting fixed seed and deterministic=True in SOT
    # configs
    if args.seed is not None:
        cfg.seed = args.seed
    elif cfg.get('seed', None) is None:
        cfg.seed = init_random_seed()
    cfg.seed = cfg.seed + dist.get_rank() if args.diff_seed else cfg.seed

    deterministic = True if args.deterministic else cfg.get(
        'deterministic', False)
    logger.info(f'Set random seed to {cfg.seed}, '
                f'deterministic: {deterministic}')
    set_random_seed(cfg.seed, deterministic=deterministic)
    meta['seed'] = cfg.seed

    # build datasets
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # build model
    if cfg.get('train_cfg', False):
        model = build_model(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    else:
        model = build_model(cfg.model)
    model.init_weights()

    # if model loaded, carry CLASSES to loaded model as it may have been trained
    # on different classes
    if args.load_from is not None:
        checkpoint = load_checkpoint(
            model, args.load_from, map_location='cpu')
        if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        logger.info(f'Checkpoint successfully loaded from {args.load_from}.')
        
    if not hasattr(model, 'CLASSES'):
        model.CLASSES = datasets[0].CLASSES

    # datasets = [build_dataset(cfg.data.train)]
    # if len(cfg.workflow) == 2:
    #     val_dataset = copy.deepcopy(cfg.data.val)
    #     val_dataset.pipeline = cfg.data.train.pipeline
    #     datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmtrack version, config file content and class names in
        # checkpoints as meta data
        cat_name_to_label = (
            datasets[0].cat_name_to_label 
            if hasattr(datasets[0], 'cat_name_to_label') else None)
        cfg.checkpoint_config.meta = dict(
            mmtrack_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            cat_name_to_label=cat_name_to_label)
    ckpt_metadata_yml = osp.join(cfg.work_dir, 'checkpoint_meta.yml')
    logger.info(f'Saving checkpoint metadata to {ckpt_metadata_yml}')
    with open(ckpt_metadata_yml, 'w') as yml_file:
        yaml.dump(
            cfg.checkpoint_config.meta, yml_file, default_flow_style=False
        )

    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    model.cat_name_to_label = cat_name_to_label

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
