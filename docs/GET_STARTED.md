## Training Details
We train all models with a total batch size of 16, split among 8 GPUs (NVIDIA RTX 2080Ti).

## Source Training (QDTrack)
Our work relies on [QDTrack](https://arxiv.org/abs/2006.06664)'s appearance-based tracker. 
You can find the config files for training a QDTrack model on several source domains in [configs/mot/qdtrack](configs/mot/qdtrack).
Source dataset configs can be found in [configs/_base_/datasets](configs/_base_/datasets).

### Training on a single GPU

```shell
python tools/run/train.py ${CONFIG_FILE} [optional arguments]
```
e.g.
```shell
python tools/run/train.py \
    configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_mot17-private-half.py \
    --work-dir ./work-dir \
    --exp-name qdtrack_bdd
```

During training, log files and checkpoints will be saved to the working directory, which is specified by `work_dir` in the config file or via CLI argument `--work-dir`.

### Training on multiple GPUs

We provide `tools/run/dist_train.sh` to launch training on multiple GPUs.
The basic usage is as follows.

```shell
bash ./tools/run/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```

For example:
```shell
JOB_NAME=qdtrack_shift_frcnn_6e
CONFIG_FILE=configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_6e_shift_track.py

bash ./tools/run/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    --exp-name ${JOB_NAME} \
    [optional arguments]
```


Optional arguments remain the same as stated above.

If you would like to launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

### Training the source models
We provide config files and pre-trained weights for QDTrack models trained on a variety of source domains. CH indicates the CrowdHuman dataset. Ped. means that the model was trained and evaluated to track pedestrians only.

|Dataset| DetA | MOTA | HOTA | IDF1 | AssA | Config | Weights |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| SHIFT | 46.9 | 48.4 |  55.2 | 60.6 | 65.8 | [config](configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_6e_shift_track.py) | [weights](https://drive.google.com/file/d/1_TMLCqEi6xSUoTxVYIpJkuufWro2FfxM/view?usp=drive_link) |
| MOT17 | 57.2 | 68.2 |  57.1 | 68.5 | 57.4 | [config](configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_mot17-private-half.py) | [weights](https://drive.google.com/file/d/11O0QjRexO78k-GquSLTRH8rVq4YQ9qni/view?usp=drive_link) |
| MOT17 (+ CH) | 59.8 | 71.7 |  59.7 | 71.6 | 58.7 | [config](configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17-private-half.py) | [weights](https://drive.google.com/file/d/1tP6dIaazHA5ViwECodgvI0s5R5Y6wVCs/view?usp=drive_link) |
| DanceTrack | 68.5 | 79.2 |  43.5 | 42.3 | 28.0 | [config](configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_dancetrack.py) | [weights](https://drive.google.com/file/d/1x_CU_EFC1GyjKDM3WnmBF5jyOzSE_JcG/view?usp=drive_link) |
| BDD100K (ped.) | 36.5 | 14.2 |  39.6 | 48.2 | 43.3 | [config](configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_12e_bdd_pedestrian_track_det.py) | [weights](https://drive.google.com/file/d/1CFhYegtJ-MPP2eh9IeUWeixEa5py1Q_O/view?usp=drive_link) |


## Test-time Adaptation (DARTH)
We present DARTH, a holistic test-time adaptation method for multiple object tracking. 
You can find the config files for adapting a QDTrack model on several target domains with DARTH in [configs/mot/qdtrack](configs/mot/qdtrack).
Target dataset configs can be found in [configs/_base_/target_datasets](configs/_base_/target_datasets). They differ from the source datasets since test-time adaptation is performed on the validation set of each dataset.

### Adapting on multiple GPUs

The basic usage is as follows, where `CHECKPOINT` is the source checkpoint trained as above.

```shell
declare -a CFG_OPTIONS=(
    "model.init_cfg.checkpoint=${CHECKPOINT}"
    "find_unused_parameters=True"
)

bash ./tools/run/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    --cfg-options ${CFG_OPTIONS[@]} \    
    [optional arguments]
```

For example:

```shell
CHECKPOINT=work_dirs/shift/qdtrack/train_qdtrack_shift_frcnn_6e/latest.pth
JOB_NAME=darth_bdd_as_shift_from_shift_frcnn_12e
CONFIG_FILE=configs/adapters/darth/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_as_shift.py

declare -a CFG_OPTIONS=(
    "data.workers_per_gpu=2"
    "data.samples_per_gpu=2"
    "log_config.hooks.1.init_kwargs.name=${JOB_NAME}"
    "find_unused_parameters=True"
    "evaluation.interval=2"
    "optimizer.lr=0.002"
    "optimizer_config.grad_clip.max_norm=35.0"
    "model.loss_rpn_distillation.loss_weight=0.1"
    "model.loss_roi_distillation.loss_weight=0.1"
    "custom_hooks.0.momentum=0.002"
    "model.init_cfg.checkpoint=${CHECKPOINT}"
    "model.conf_thr=0.7"
)

bash ./tools/run/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    --exp-name ${JOB_NAME} \
    --cfg-options ${CFG_OPTIONS[@]}
```

### Adapting to the target domain
We provide config files and pre-trained weights for DARTH models adapted to a variety of target domains. We report the performance on the target dataset after adaptation. CH indicates the CrowdHuman dataset. Ped. means that the model was trained and evaluated to track pedestrians only.

| Source | Target | DetA | MOTA | HOTA | IDF1 | AssA | Config | Weights |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| SHIFT | BDD100K | 15.2 | 8.3 |  20.6 | 23.7 | 33.1 | [config](configs/adapters/darth/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_as_shift.py) | [weights](https://drive.google.com/file/d/1tnM8gmWR-sE-ZQpwmrX3MscGV_Ttr7lB/view?usp=drive_link) |
| MOT17 | DanceTrack | 57.2 | 70.1 |  31.6 | 32.8 | 17.7 | [config](configs/adapters/darth/darth_qdtrack_faster-rcnn_r50_fpn_4e_dancetrack.py) | [weights](https://drive.google.com/file/d/1YDGm-RkOqqv1D8AdSU02hYGm1W9POst8/view?usp=drive_link) |
| MOT17 | BDD100K | 31.6 | 21.4 |  32.4 | 40.4 | 33.6 | [config](configs/adapters/darth/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_pedestrian_from_mot.py) | [weights](https://drive.google.com/file/d/1K1VygKYnIUu6C2i5opdoYRVLh5Cv4c_M/view?usp=drive_link) |
| MOT17 (+ CH) | DanceTrack | 64.7 | 78.9 |  35.4 | 35.3 | 19.6 | [config](configs/adapters/darth/darth_qdtrack_faster-rcnn_r50_fpn_4e_dancetrack.py) | [weights](https://drive.google.com/file/d/1ZJ5eAf6xQVsHOsaTRLjkPE1tt2c_xOXi/view?usp=drive_link) |
| MOT17 (+ CH) | BDD100K | 36.3 | 23.4 |  36.3 | 44.4 | 36.8 | [config](configs/adapters/darth/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_pedestrian_from_mot.py) | [weights](https://drive.google.com/file/d/1H2cSpD7zwDbCPQNMc5A0OxZQffDklgAs/view?usp=drive_link) |
| DanceTrack | MOT17 | 26.4 | 25.5 |  34.3 | 37.9 | 45.2 | [config](configs/adapters/darth/darth_qdtrack_faster-rcnn_r50_fpn_4e_mot17-private-half.py) | [weights](https://drive.google.com/file/d/1d8rWkb9Qa7Pv0zOcJ1VCvq8kXrJ4SOF5/view?usp=drive_link) |
| DanceTrack | BDD100K (Ped.) | 12.8 | -1.5 |  17.8 | 17.4 | 25.1 | [config](configs/adapters/darth/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_pedestrian_from_mot.py) | [weights](https://drive.google.com/file/d/1ShgX0BgtwIjjL5J1TBngswUHOUiDVTgz/view?usp=drive_link) |
| BDD100K (ped.) | MOT17 | 29.4 | 32.6 |  36.6 | 44.4 | 45.9 | [config](configs/adapters/darth/darth_qdtrack_faster-rcnn_r50_fpn_4e_mot17-private-half_from_bdd.py) | [weights](https://drive.google.com/file/d/1TRhXYG0-0kypUWV_wRMvIlsHnbDTHL4C/view?usp=drive_link) |
| BDD100K (ped.) | DanceTrack | 45.1 | 50.2 |  21.5 | 21.4 | 10.4 | [config](configs/adapters/darth/darth_qdtrack_faster-rcnn_r50_fpn_4e_dancetrack_from_bdd.py) | [weights](https://drive.google.com/file/d/1rhL3jzCRzD-BLfPXnXBEamNuX3-pFthE/view?usp=drive_link) |
