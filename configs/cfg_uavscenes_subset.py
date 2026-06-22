_base_ = './cfg_uavscenes.py'

# override: use 500-image subset for quick validation
test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='/media/zero/DAAA8605AA85DE7F1/SegEarth-OV-2/uavscenes_val_subset/images',
            seg_map_path='/media/zero/DAAA8605AA85DE7F1/SegEarth-OV-2/uavscenes_val_subset/labels'),
    )
)
