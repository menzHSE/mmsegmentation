_base_ = './pspnet_r50-d8_4xb4-80k_stihl-486x648.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(type='ResNet', depth=101))
