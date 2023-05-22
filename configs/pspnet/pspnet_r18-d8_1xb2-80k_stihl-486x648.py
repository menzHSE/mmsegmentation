_base_ = './pspnet_r50-d8_1xb2-80k_stihl-486x648.py'
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        num_classes=23
    ),
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=23))
