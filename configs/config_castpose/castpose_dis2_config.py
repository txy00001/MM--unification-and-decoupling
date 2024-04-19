_base_ = [
    '/home/txy/code/TXY_code/work_dirs/castpose/castpose_config.py'  # noqa: E501
]

# model settings
find_unused_parameters = True

# dis settings
second_dis = True

# config settings
logit = True

train_cfg = dict(max_epochs=150, val_interval=10)

# method details
model = dict(
    _delete_=True,
    type='DWPoseDistiller',
    two_dis=second_dis,
    teacher_pretrained='/home/txy/code/TXY_code/work_dirs/castpose/epoch_270.pth',  # noqa: E501
    teacher_cfg='/home/txy/code/TXY_code/work_dirs/castpose/castpose_config.py',  # noqa: E501
    student_cfg='/home/txy/code/TXY_code/work_dirs/castpose/castpose_config.py',  # noqa: E501
    distill_cfg=[
        dict(methods=[
            dict(
                type='KDLoss',
                name='loss_logit',
                use_this=logit,
                weight=1,
            )
        ]),
    ],
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    train_cfg=train_cfg,
)

optim_wrapper = dict(clip_grad=dict(max_norm=1., norm_type=2))