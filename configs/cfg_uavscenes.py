_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_uavscenes.txt',
    prob_thd=0.0,          # keep more predictions (19 fine-grained classes)
)

# dataset settings
dataset_type = 'UAVScenesDataset'
data_root = ''

# UAVScenes label remap table (0-25 raw → 0-18 effective + 255 ignore)
uavscenes_label_map = [
    0, 1, 2, 3, 4, 5, 6, 255, 255, 7,
    8, 9, 255, 10, 11, 12, 13, 14, 15, 16,
    17, 255, 255, 255, 18, 255,
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='RemapLabels', mapping=uavscenes_label_map),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='/media/zero/DAAA8605AA85DE7F1/SegEarth-OV-2/uavscenes_val/images',
            seg_map_path='/media/zero/DAAA8605AA85DE7F1/SegEarth-OV-2/uavscenes_val/labels'),
        pipeline=test_pipeline))
