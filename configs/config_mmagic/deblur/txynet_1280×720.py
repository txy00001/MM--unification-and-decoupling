_base_ = '/home/txy/code/TXY_code/configs/config_mmagic/_base_/default_runtime.py'

experiment_name = 'TXYfnet_416_416  '
work_dir = f'/home/txy/code/TXY_code/work_dirs/deblur_TXYNet/{experiment_name}'
save_dir = '/home/txy/code/TXY_code/work_dirs/deblur_TXYNet/'

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='TXYNetLocal',
        img_channels=3,
        mid_channels=32,
        enc_blk_nums=[3, 3, 3],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1],
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    cleaning_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    is_use_sharpened_gt_in_pixel=True,
    is_use_ema=True,
    train_cfg=dict(fix_iter=5000),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=4)),
    dict(type='FixedCrop', keys=['gt'], crop_size=(256, 256)),
    dict(type='Flip', keys=['gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['gt'], transpose_ratio=0.5),
    dict(type='MirrorSequence', keys=['gt']),
    dict(
        type='UnsharpMasking',
        keys=['gt'],
        kernel_size=51,
        sigma=0,
        weight=0.5,
        threshold=10),
    dict(type='CopyValues', src_keys=['gt_unsharp'], dst_keys=['img']),
    dict(
        type='RandomBlur',
        params=dict(
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 3],
            sigma_y=[0.2, 3],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2],
            sigma_x_step=0.02,
            sigma_y_step=0.02,
            rotate_angle_step=0.31416,
            beta_gaussian_step=0.05,
            beta_plateau_step=0.1,
            omega_step=0.0628),
        keys=['img'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0.2, 0.7, 0.1],  # up, down, keep
            resize_scale=[0.15, 1.5],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3.0, 1 / 3.0, 1 / 3.0],
            resize_step=0.015,
            is_size_even=True),
        keys=['img'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 30],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 3],
            poisson_gray_noise_prob=0.4,
            gaussian_sigma_step=0.1,
            poisson_scale_step=0.005),
        keys=['img'],
    ),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[30, 95], quality_step=3),
        keys=['img'],
    ),
    dict(
        type='RandomVideoCompression',
        params=dict(
            codec=['libx264', 'h264', 'mpeg4'],
            codec_prob=[1 / 3., 1 / 3., 1 / 3.],
            bitrate=[1e4, 1e5]),
        keys=['img'],
    ),
    dict(
        type='RandomBlur',
        params=dict(
            prob=0.8,
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 1.5],
            sigma_y=[0.2, 1.5],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2],
            sigma_x_step=0.02,
            sigma_y_step=0.02,
            rotate_angle_step=0.31416,
            beta_gaussian_step=0.05,
            beta_plateau_step=0.1,
            omega_step=0.0628),
        keys=['img'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0.3, 0.4, 0.3],  # up, down, keep
            resize_scale=[0.3, 1.2],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.],
            resize_step=0.03,
            is_size_even=True),
        keys=['img'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 25],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 2.5],
            poisson_gray_noise_prob=0.4,
            gaussian_sigma_step=0.1,
            poisson_scale_step=0.005),
        keys=['img'],
    ),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[30, 95], quality_step=3),
        keys=['img'],
    ),
    dict(
        type='DegradationsWithShuffle',
        degradations=[
            dict(
                type='RandomVideoCompression',
                params=dict(
                    codec=['libx264', 'h264', 'mpeg4'],
                    codec_prob=[1 / 3., 1 / 3., 1 / 3.],
                    bitrate=[1e4, 1e5]),
                keys=['img'],
            ),
            [
                dict(
                    type='RandomResize',
                    params=dict(
                        target_size=(64, 64),
                        resize_opt=['bilinear', 'area', 'bicubic'],
                        resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
                ),
                dict(
                    type='RandomBlur',
                    params=dict(
                        prob=0.8,
                        kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
                        kernel_list=['sinc'],
                        kernel_prob=[1],
                        omega=[3.1416 / 3, 3.1416],
                        omega_step=0.0628),
                ),
            ]
        ],
        keys=['img'],
    ),
    dict(type='Clip', keys=['img']),
    dict(type='PackInputs')
]
val_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:04d}.png'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]

test_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:08d}.png'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='PackInputs')
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='PackInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'

train_dataloader = dict(
    num_workers=2,
    batch_size=16,  # gpus 4
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='reds', task_name='vsr'),
        data_root='/mnt/P40_NFS/20_Research/10_公共数据集/40_DeBlur/ALL/',
        data_prefix=dict(gt='sharp', img='blur'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='gopro', task_name='deblur'),
        data_root='/mnt/P40_NFS/20_Research/10_公共数据集/40_DeBlur/ALL/',
        data_prefix=dict(gt='sharp', img='blur'),
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR'),
    dict(type='SSIM'),
]
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=400000, val_interval=20000)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.9)))

# learning policy
param_scheduler = dict(
    type='CosineAnnealingLR', by_epoch=False, T_max=400_000, eta_min=1e-7)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=3000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

randomness = dict(seed=10, diff_rank_seed=True)
