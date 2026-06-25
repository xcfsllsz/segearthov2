_base_ = './cfg_uavscenes.py'

dataset_type = 'UAVScenes11cDataset'

# 11-class remap: dirt road(2), pool(5), bridge(6), traffic barrier(11),
#   umbrella(16), glass roof(17), sidewalk(19), truck(24) → 255
label_map_11c = [
    0, 1, 255, 2, 3, 255, 255, 255, 255, 4,
    5, 255, 255, 6, 7, 8, 255, 255, 9, 255,
    10, 255, 255, 255, 255, 255,
]

model = dict(
    name_path='./configs/cls_uavscenes_11c.txt',
    prob_thd=0.10,
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='RemapLabels', mapping=label_map_11c),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_prefix=dict(
            img_path='/media/zero/DAAA8605AA85DE7F1/SegEarth-OV-2/uavscenes_val_1k/images',
            seg_map_path='/media/zero/DAAA8605AA85DE7F1/SegEarth-OV-2/uavscenes_val_1k/labels'),
    )
)
