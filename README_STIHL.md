# STIHL SemSeg

## Environment
* Clone this repo
* Create a conda environment, install PyTorch and install mmsegmentation via the [official instructions](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)
* During the *Install MMSegmentation* step in the [official instructions](https://mmsegmentation.readthedocs.io/en/latest/get_started.html), replace the original mmsegmentation repo `https://github.com/open-mmlab/mmsegmentation.git` by your clone of this repo 
* Verfied versions: local installation Ubuntu 20.04.3 LTS (PyTorch 1.13.1 with CUDA 11.6 and 12 GB Geforce Titan X); bwUniCluster (PyTorch 1.13.1 with CUDA 11.6 and P100/A100 GPUS)

## Prepare the Dataset
* Link the STIHL dataset to mmsegmentation/data/STIHL_SemSeg
* Run ```tools/dataset_converters/stihl.py``` on the ```color_mask``` images to convert them from RGB to grayscale with the class id being used as pixel intensity value. Save all converted images in a folder called ```gray_mask```
* Use the following directory structure to split between annotations (```gray_mask```) / input images and train / val

```
data/STIHL_SemSeg
├── ann_dir
│   ├── train
│   └── val
└── img_dir
    ├── train
    └── val
``` 

## Train and Test

### Local 
There are several configs available with a ```_stihl``` suffix, e.g. in ```pspnet``` or ```pidnet```. These can be trained / tested using the regular mmsegmentation ```train.py``` scripts:

```
python tools/train.py configs/pspnet/pspnet_r18-d8_1xb2-80k_stihl-486x648.py  --resume

python tools/test.py configs/pspnet/pspnet_r101-stihl.py work_dirs/pspnet_r101-stihl/iter_80000.pth --show-dir work_dirs/pspnet_r101-stihl/test_iter_80000
```

### bwUniCluster
#### Preparation / Tips
* login via SSH and start a new tmux session: ```tmux```. You can then close the connection to bwUniCluster anytime and resume it via attaching to the tmux session, e.g. ```tmux a```. ```tmux ls``` shows all existing active sessions. [Learn to use tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/)
* `dist_train.sh` can easily be used for multi-GPU training on bwUniCluster, as follows:

#### (Distributed) Training

* Create a shell script containing conda environment activation and the training command, e.g.
```
#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate mmsegmentation-pyTorch-1.13.1-cuda-11.6
cd /home/es/es_es/es_menzweil/Development/mmsegmentation

# pspnet resnet 101 backbone, 4 GPUs
sh tools/dist_train.sh configs/pspnet/pspnet_r101-stihl.py 4
```
* Allocate resources on a `dev_gpu` node to try-out / debug your bash script
```
salloc -p dev_gpu_4 -t 00:30:00 -n 20 --gres=gpu:4

# run your script 
```
* Submit your job
```
sbatch -p gpu_4_a100 -t 24:00:00 -n 1 --gres=gpu:4 $HOME/Development/mmsegmentation/trainStihl_r101_4gpu.bash
```
* Log files can be found in `mmsegmentation/work_dirs/<YOUR EXPERIMENT>` and/or in a SLURM log file  `slurm-JOBID.out` in your home directory on bwUniCluster. 

