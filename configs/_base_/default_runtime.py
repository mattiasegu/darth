# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(interval=1)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook', by_epoch=False,
             # The Wandb logger is also supported, It requires `wandb` to be installed.
             init_kwargs={
                'project': "darth",
                'name': None,
            }),  # Check https://docs.wandb.ai/ref/python/init for more init arguments.
    ])
# yapf:enable
# dist_params = dict(backend='nccl', port=29500)
dist_params = dict(backend='nccl')
find_unused_parameters = False
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'