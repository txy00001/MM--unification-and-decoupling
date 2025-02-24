

_base_ = ['/home/txy/code/CastPose/configs/_base_/default_runtime.py']
max_epochs = 300
stage2_num_epochs = 10
base_lr = 0.0001

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Lion', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),##梯度裁剪
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(288, 384),
    sigma=(6., 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='/home/txy/code/CastPose/work_dirs/wholebody_impove/best_coco_AP_epoch_40.pth'  # noqa
        )),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=None,
        out_indices=(
            1,
            2,
        ),
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='neck.',
            checkpoint='/home/txy/code/CastPose/work_dirs/wholebody_impove/best_coco_AP_epoch_40.pth'  # noqa
        )
        ),
    ##加入可见性预测
    head=dict(
        type='VisPredictHead',
        loss=dict(
               type='BCELoss',
               use_target_weight=True,
               use_sigmoid=True,
               loss_weight=1e-3),
        pose_cfg=dict(    
            type='RTMWHead',
            in_channels=1024,
            out_channels=44,
            input_size=(288,384),
            in_featuremap_size=(9,12),
            simcc_split_ratio=codec['simcc_split_ratio'],
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False),
            loss=dict(
                type='KLDiscretLoss',
                use_target_weight=True,
                beta=10.,
                label_softmax=True),
            decoder=codec),
            ),
    test_cfg=dict(flip_test=True))
# base dataset settings

data_mode = 'topdown'


backend_args = dict(backend='local')


# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5],
        rotate_factor=180),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=180),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]


#数据集加载合并
##将cocowholebody转为我们自己的数据集
dataset_wholebody = dict(
    type='CocoWholeBodyDataset',
    data_mode=data_mode,
    data_root='/mnt/P40_NFS/',
    ann_file='20_Research/10_公共数据集/10_Pose/coco/coco_wholebody_train_v1.0.json',
    data_prefix=dict(img='train2017/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=44,  # 与 我们的 数据集关键点数一致
            mapping=[  # 需要列出所有带转换关键点的序号
                (0, 0),  
                (1, 3),
                (2, 4),
                (3, 5),
                (4, 6),
                (5, 7),
                (6, 8),
                (7, 9),
                (8, 10),
                (9, 11),
                (10, 12),
                (11, 13),  # 91 (wholebody 中的序号) -> 11 (我们数据集 中的序号)
                (12, 14),
                (13, 15),
                (14, 16),
                (15, 17),
                (16, 18),
                (17, 19),
                (18, 20),
                (19, 21),
                (20, 22),
                
            ])
    ],
)


dataset_coco1=dict(
    type='QXCastHandDatasets',
    data_root='/home/txy/data/qx_data/casthand/',
    data_mode=data_mode,
    ann_file='hand_826_1/train_coco_hand_8.26.json',
    data_prefix=dict(img='images/'),
    pipeline=[
         dict(
            type='KeypointConverter',
            num_keypoints=44,  
            mapping=[(0, 0),  
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (10, 10),
                (11, 11),  # 91 (wholebody 中的序号) -> 11 (我们数据集 中的序号)
                (12, 12),
                (13, 13),
                (14, 14),
                (15, 15),
                (16, 16),
                (17, 17),
                (18, 18),
                (19, 19),
                (20, 20),
                (21, 21),
                (22, 22),
                (23,23),
                (24,24),
                (25, 25),
                (26, 26),
                (27, 27),
                (28, 28),
                (29, 29),
                (30, 30),
                (31, 31),
                (32, 32),
                (33, 33),
                (34, 34),
                (35, 35),
                (36, 36),
                (37, 37),
                (38, 38),
                (39, 39),
                (40, 40),
                (41, 41),
                (42, 42),
                (43, 43),
                
                ],####我们自己的数据集不需要转换
)
    ],
)

dataset_coco2=dict(
    type='QXCastHandDatasets',
    data_root='/home/txy/data/qx_data/casthand/',
    data_mode=data_mode,
    ann_file='hand_826_2/train_coco_hand_8.26.json',
    data_prefix=dict(img='images_20230826/'),
    pipeline=[
         dict(
            type='KeypointConverter',
            num_keypoints=44,  
            mapping=[(0, 0),  
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (10, 10),
                (11, 11),  # 91 (wholebody 中的序号) -> 11 (我们数据集 中的序号)
                (12, 12),
                (13, 13),
                (14, 14),
                (15, 15),
                (16, 16),
                (17, 17),
                (18, 18),
                (19, 19),
                (20, 20),
                (21, 21),
                (22, 22),
                (23,23),
                (24,24),
                (25, 25),
                (26, 26),
                (27, 27),
                (28, 28),
                (29, 29),
                (30, 30),
                (31, 31),
                (32, 32),
                (33, 33),
                (34, 34),
                (35, 35),
                (36, 36),
                (37, 37),
                (38, 38),
                (39, 39),
                (40, 40),
                (41, 41),
                (42, 42),
                (43, 43),
                
                ],####我们自己的数据集不需要转换
)
    ],
)

dataset_coco3=dict(
    type='QXCastHandDatasets',
    data_root='/home/txy/data/qx_data/casthand/',
    data_mode=data_mode,
    ann_file='hand_1026_1/train_coco_hand_10.26.json',
    data_prefix=dict(img='images_20231026/'),
   pipeline=[
         dict(
            type='KeypointConverter',
            num_keypoints=44,  
            mapping=[(0, 0),  
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (10, 10),
                (11, 11),  # 91 (wholebody 中的序号) -> 11 (我们数据集 中的序号)
                (12, 12),
                (13, 13),
                (14, 14),
                (15, 15),
                (16, 16),
                (17, 17),
                (18, 18),
                (19, 19),
                (20, 20),
                (21, 21),
                (22, 22),
                (23,23),
                (24,24),
                (25, 25),
                (26, 26),
                (27, 27),
                (28, 28),
                (29, 29),
                (30, 30),
                (31, 31),
                (32, 32),
                (33, 33),
                (34, 34),
                (35, 35),
                (36, 36),
                (37, 37),
                (38, 38),
                (39, 39),
                (40, 40),
                (41, 41),
                (42, 42),
                (43, 43),
                
                ],####我们自己的数据集不需要转换
)
    ],
)
dataset_coco4=dict(
    type='QXCastHandDatasets',
    data_root='/home/txy/data/qx_data/casthand/',
    data_mode=data_mode,
    ann_file='hand_1026_2/train_coco_hand_10.26.json',
    data_prefix=dict(img='images_20231026/'),
    pipeline=[
         dict(
            type='KeypointConverter',
            num_keypoints=44,  
            mapping=[(0, 0),  
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (10, 10),
                (11, 11),  # 91 (wholebody 中的序号) -> 11 (我们数据集 中的序号)
                (12, 12),
                (13, 13),
                (14, 14),
                (15, 15),
                (16, 16),
                (17, 17),
                (18, 18),
                (19, 19),
                (20, 20),
                (21, 21),
                (22, 22),
                (23,23),
                (24,24),
                (25, 25),
                (26, 26),
                (27, 27),
                (28, 28),
                (29, 29),
                (30, 30),
                (31, 31),
                (32, 32),
                (33, 33),
                (34, 34),
                (35, 35),
                (36, 36),
                (37, 37),
                (38, 38),
                (39, 39),
                (40, 40),
                (41, 41),
                (42, 42),
                (43, 43),
                
                ],####我们自己的数据集不需要转换
)
    ],
)


###val数据集
dataset_coco_val=dict(
    type='QXCastHandDatasets',
    data_root='/home/txy/data/qx_data/casthand/',
    data_mode=data_mode,
    ann_file='hand_826_2/val_coco_hand_8.26.json',
    data_prefix=dict(img='images/'),
    pipeline=[
         dict(
            type='KeypointConverter',
            num_keypoints=44,  
            mapping=[(0, 0),  
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (10, 10),
                (11, 11),  # 91 (wholebody 中的序号) -> 11 (我们数据集 中的序号)
                (12, 12),
                (13, 13),
                (14, 14),
                (15, 15),
                (16, 16),
                (17, 17),
                (18, 18),
                (19, 19),
                (20, 20),
                (21, 21),
                (22, 22),
                (23,23),
                (24,24),
                (25, 25),
                (26, 26),
                (27, 27),
                (28, 28),
                (29, 29),
                (30, 30),
                (31, 31),
                (32, 32),
                (33, 33),
                (34, 34),
                (35, 35),
                (36, 36),
                (37, 37),
                (38, 38),
                (39, 39),
                (40, 40),
                (41, 41),
                (42, 42),
                (43, 43),
                
                ],####我们自己的数据集不需要转换
)
    ],
)


# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='/home/txy/code/TXY_code/config_castpose/_base_/datasets/casthand.py'),
        datasets=[dataset_coco1,
                  dataset_coco2, 
                  dataset_coco3,
                  dataset_coco4,
                  ],
        pipeline=train_pipeline,
        # sample_ratio_factor=[2, 1],
        test_mode=False,
    ))



val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
         metainfo=dict(from_file='/home/txy/code/TXY_code/config_castpose/_base_/datasets/casthand.py'),
        datasets=[dataset_coco_val,
                  
                  ],
        pipeline=val_pipeline,
        # sample_ratio_factor=[2, 1],
        test_mode=False,
    ))
test_dataloader = val_dataloader

# hooks
# hooks
default_hooks = dict(
    checkpoint=dict(save_best='AUC', rule='greater', max_keep_ckpts=100))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = [
    dict(type='PCKAccuracy', thr=0.2),
    dict(type='AUC'),
    dict(type='EPE')
]
test_evaluator = val_evaluator
