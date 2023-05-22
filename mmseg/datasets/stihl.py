# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class STIHLDataset(BaseSegDataset):
    """STIHL dataset. """
    METAINFO = dict(
        classes=('Unlabeled', 'Ego','NaturalGround','Boundary:Wall','Boundary:Fence','Boundary:Hedge','Boundary:Building','GenericObject','Vegetation','Lawn','Sky','DockingStation','DockingPad','ArtificialGround','ArtificialFlatObject','iMOW','Tool','Human:Adult','Human:Child','Distortion','NaturalFlatObject','Animal:Dog','Furniture'),
        palette=[[0,0,0],[255,255,0],[0,255,255],[64,64,64],[148,94,37],[78,148,37],[78,148,10],[0,64,255],[0,128,0],[0,255,0],[0,128,255],[255,0,0],[128,0,0],[128,128,128],[255,0,128],[255,128,0],[64,64,0],[255,10,10],[255,20,20],[200,200,200],[15,255,30],[255,128,128],[200,200,200]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 ignore_index=0, #ignore Unlabeled
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
