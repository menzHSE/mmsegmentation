# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import shutil
import tempfile
import zipfile

import mmcv
import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist
from PIL import Image

stihl_palette = \
    {
        0: (0,0,0),
        1: (0,255,255),
        2: (255,255,0),
        3: (64,64,64),
        4: (37,94,148),
        5: (37,148,78), 
        6: (10,148,78),
        7: (255,64,0),
        8: (0,128,0),
        9: (0,255,0),
        10: (255,128,0),
        11: (0,0,255),
        12: (0,0,128),
        13: (128,128,128),
        14: (128,0,255),
        15: (0,128,255),
        16: (0,64,64),
        17: (10,10,255),
        18: (20,20,255),
        19: (200,200,200),
        20: (30,255,15),
        21: (128,128,255),
        22: (200,200,200) 
    }

stihl_invert_palette = {v: k for k, v in stihl_palette.items()}


def stihl_convert_from_color(arr_3d, palette=stihl_invert_palette):
    """RGB-color encoding to grayscale labels."""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        #print(f"{c[::-1]} --> {i}")
        m = np.all(arr_3d == np.array(c[::-1]).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d




def convert_label(src_path, out_dir):
    label = mmcv.imread(src_path, channel_order='rgb')
    label = stihl_convert_from_color(label)
    image = osp.basename(src_path)
    out_path = osp.join(out_dir, image)
    print(f'Converting {src_path} to {out_path}')
    outLabel = Image.fromarray(label.astype(np.uint8), mode='P')
    outLabel.save(out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert STIHL dataset to mmsegmentation format')
    parser.add_argument('annotation_folder', help='STIHL annotation folder path')
    parser.add_argument('-o', '--out_dir', help='output path')

    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.annotation_folder


    if args.out_dir is None:
        out_dir = osp.join('data', 'STIHL')
    else:
        out_dir = args.out_dir


    src_path_list = glob.glob(os.path.join(dataset_path, '*.png'))

            
    for i, img_path in enumerate(src_path_list):
        convert_label(img_path, out_dir)

    print('Done!')


if __name__ == '__main__':
    main()
