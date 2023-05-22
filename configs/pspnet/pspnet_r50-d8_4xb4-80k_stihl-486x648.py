_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/stihl.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k_stihl.py'
]
crop_size = (486, 648)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=23),
    auxiliary_head=dict(num_classes=23))
