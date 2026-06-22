import os.path as osp
import numpy as np
import mmengine.fileio as fileio

from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class OpenEarthMapDataset(BaseSegDataset):
    """OpenEarthMap dataset.

    In segmentation map annotation for OpenEarthMap, 0 is the ignore index.
    ``reduce_zero_label`` should be set to False. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.tif'.
    """
    METAINFO = dict(
        classes=('background', 'bareland', 'grass', 'pavement', 'road', 'tree',
                 'water', 'cropland', 'building'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 255, 36], [148, 148, 148],
                 [255, 255, 255], [34, 97, 38], [0, 69, 255], [75, 181, 73],
                 [222, 31, 7]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class WHUDataset(BaseSegDataset):
    """WHU dataset.

    """
    METAINFO = dict(
        classes=('background', 'building'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class xBDDataset(BaseSegDataset):
    """xBD dataset.

    """
    METAINFO = dict(
        classes=('background', 'building'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class CHN6_CUGDataset(BaseSegDataset):
    """CHN6-CUG dataset.

    """
    METAINFO = dict(
        classes=('background', 'road'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class RoadValDataset(BaseSegDataset):
    """RoadVal dataset.

    """
    METAINFO = dict(
        classes=('background', 'road'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class UAVidDataset(BaseSegDataset):
    """UAVid dataset.

    convert Moving_Car to Static_Car
    """
    METAINFO = dict(
    classes=('background', 'building', 'road', 'car', 'tree', 
             'vegetation', 'human'),
    palette=[[0, 0, 0], [128, 0, 0], [128, 64, 128], [192, 0, 192], 
             [0, 128, 0], [128, 128, 0], [64, 64, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)


@DATASETS.register_module()
class UDD5Dataset(BaseSegDataset):
    """UDD5 dataset.
    
    """
    METAINFO = dict(
    classes=('vegetation', 'building', 'road', 'vehicle',
             'other'),
    palette=[[107, 142, 35], [102,102,156], [128,64,128],
             [0, 0, 142], [0, 0, 0]])

    def __init__(self,
                 img_suffix='.JPG',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)


@DATASETS.register_module()
class VDDDataset(BaseSegDataset):
    """VDD dataset.
    
    """
    METAINFO = dict(
    classes=('other', 'wall', 'road', 'vegetation', 'vehicle',
             'roof', 'water'))

    def __init__(self,
                 img_suffix='.JPG',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)


@DATASETS.register_module()
class InriaDataset(BaseSegDataset):
    """Inria dataset.

    """
    METAINFO = dict(
        classes=('background', 'building'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class WaterDataset(BaseSegDataset):
    """Water dataset.

    """
    METAINFO = dict(
        classes=('background', 'water'),
        palette=[[0, 0, 0], [0, 235, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.jpg',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class PIESARDataset(BaseSegDataset):
    """PIE-SAR dataset.

    """
    METAINFO = dict(
        classes=('background', 'city', 'road', 'water', 'forest', 'farmland'),
        palette=[[0, 0, 0], [222, 31, 7], [255, 255, 255], [0, 69, 255],
                 [34, 97, 38], [75, 181, 73]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class WHUSARDataset(BaseSegDataset):
    """WHU-SAR dataset.

    """
    METAINFO = dict(
        classes=('farmland', 'city', 'village', 'water', 'forest', 'road', 'background'),
        palette=[[75, 181, 73], [222, 31, 7], [126, 0, 30], [0, 69, 255],
                 [34, 97, 38], [255, 255, 255], [0, 0, 0]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class DDHR_Korea_SARDataset(BaseSegDataset):
    """DDHR-SAR dataset.

    """
    METAINFO = dict(
        classes=('building', 'road', 'greenery', 'water', 'farmland'),
        palette=[[222, 31, 7], [255, 255, 255], [34, 97, 38],
                 [0, 69, 255], [75, 181, 73]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class DDHR_Shandong_SARDataset(BaseSegDataset):
    """DDHR-SAR dataset.

    """
    METAINFO = dict(
        classes=('farmland', 'greenery', 'road', 'building', 'water'),
        palette=[[75, 181, 73], [34, 97, 38], [255, 255, 255],
                 [222, 31, 7], [0, 69, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class DDHR_Xian_SARDataset(BaseSegDataset):
    """DDHR-SAR dataset.

    """
    METAINFO = dict(
        classes=('building', 'road', 'farmland', 'greenery', 'water'),
        palette=[[222, 31, 7], [255, 255, 255], [75, 181, 73],
                 [34, 97, 38], [0, 69, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class FUSARDataset(BaseSegDataset):
    """FUSAR dataset.

    """
    METAINFO = dict(
        classes=('background', 'water', 'road', 'building', 'vegetation'),
        palette=[[0, 0, 0], [0, 69, 255], [255, 255, 255],
                 [222, 31, 7], [0, 255, 36]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class YESegSARDataset(BaseSegDataset):
    """YESegSAR dataset.

    """
    METAINFO = dict(
        classes=('bareground', 'grass', 'tree', 'house',
                 'water', 'road'),
        palette=[[0, 0, 0], [0, 255, 36], [34, 97, 38],
                 [222, 31, 7], [0, 69, 255], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


# ═══════════════════════════════════════════════════════
# Label remapping transform  (index=255 → ignore)
# ═══════════════════════════════════════════════════════
@TRANSFORMS.register_module()
class RemapLabels:
    """Remap label ids via lookup table.  mapping[i] = new_id."""
    def __init__(self, mapping):
        self.mapping = np.array(mapping, dtype=np.uint8)

    def __call__(self, results):
        seg_map = results['gt_seg_map']
        if hasattr(seg_map, 'numpy'):
            seg_map = seg_map.numpy()
        results['gt_seg_map'] = self.mapping[seg_map]
        return results


# UAVScenes label remap table  (cmap.py 0-25 → 19 effective classes)
#   orig:  0  1  2  3  4  5  6  7   8   9 10 11 12  13 14 15 16 17 18 19 20 21  22  23  24 25
#   new:   0  1  2  3  4  5  6 255 255  7  8  9 255 10 11 12 13 14 15 16 17 255 255 255 18 255
UAVSCENES_LABEL_MAP = [
    0, 1, 2, 3, 4, 5, 6, 255, 255, 7,
    8, 9, 255, 10, 11, 12, 13, 14, 15, 16,
    17, 255, 255, 255, 18, 255,
]

# ═══════════════════════════════════════════════════════
# UAVScenes dataset  (ICCV 2025, 19 effective classes)
#   images: .jpg   labels: .png   labels remapped via RemapLabels
# ═══════════════════════════════════════════════════════
@DATASETS.register_module()
class UAVScenesDataset(BaseSegDataset):
    METAINFO = dict(
        classes=(
            'background', 'roof', 'dirt road', 'paved road',
            'river', 'pool', 'bridge', 'container', 'airstrip',
            'traffic barrier', 'green field', 'wild field',
            'solar panel', 'umbrella', 'glass roof',
            'parking lot', 'sidewalk', 'car', 'truck',
        ),
        palette=[
            [0, 0, 0], [119, 11, 32], [180, 165, 180], [128, 64, 128],
            [173, 216, 230], [0, 80, 100], [150, 100, 100], [250, 170, 30],
            [81, 0, 81], [102, 102, 156], [107, 142, 35], [210, 180, 140],
            [220, 220, 0], [153, 153, 153], [0, 0, 90], [250, 170, 160],
            [244, 35, 232], [0, 0, 142], [0, 0, 70],
        ])

    def __init__(self, img_suffix='.jpg', seg_map_suffix='.png',
                 reduce_zero_label=False, **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix,
                         reduce_zero_label=reduce_zero_label, **kwargs)