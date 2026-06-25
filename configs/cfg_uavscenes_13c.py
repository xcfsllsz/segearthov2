_base_ = './cfg_uavscenes.py'

dataset_type = 'UAVScenes13cDataset'

# 13-class remap: dirt road(2), pool(5), bridge(6), traffic barrier(11),
#                umbrella(16), truck(24) → 255 ignore
label_map_13c = [
    0, 1, 255, 2, 3, 255, 255, 255, 255, 4,
    5, 255, 255, 6, 7, 8, 255, 9, 10, 11,
    12, 255, 255, 255, 255, 255,
]

model = dict(
    name_path='./configs/cls_uavscenes_13c.txt',
    prob_thd=0.10,
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='RemapLabels', mapping=label_map_13c),
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

# 13-class evaluator
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
