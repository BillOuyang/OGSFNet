# Occlusion Guided Scene Flow Estimation on 3D Point Clouds
This is official implementation for the paper "Occlusion Guided Scene Flow Estimation on 3D Point Clouds"


## Citation
If you find our work useful in your research, please cite:

        @InProceedings{Ouyang_2021_CVPR,
          author    = {Ouyang, Bojun and Raviv, Dan},
          title     = {Occlusion Guided Scene Flow Estimation on 3D Point Clouds},
          booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
          month     = {June},
          year      = {2021},
          pages     = {2805-2814}
        }




## Requirement
To run our model, please install the following package (we suggest to use the Anaconda environment):
* Python 3.6+
* PyTorch==1.6.0
* CUDA CuDNN
* scipy
* numpy
* tqdm

Compile the furthest point sampling, grouping and gathering operation for PyTorch. We use the operation from this [repo](https://github.com/sshaoshuai/Pointnet2.PyTorch).
```shell
cd pointnet2
python setup.py install
cd ../
```

## Data preperation
We use the Flyingthings3D and KITTI dataset preprocessed by [this work](https://github.com/xingyul/flownet3d).
Download the Flyingthings3D dataset from [here](https://drive.google.com/file/d/1CMaxdt-Tg1Wct8v8eGNwuT7qRSIyJPY-/view?usp=sharing) and KITTI dataset from [here](https://drive.google.com/open?id=1XBsF35wKY0rmaL7x7grD_evvKCAccbKi).
 Create a folder named `datasets` under the root folder. After the downloading, extract the files into the `datasets`. The directory of the datasets should looks like the following:

```
datasets/data_processed_maxcut_35_20k_2k_8192   % FlyingThings3D dataset
datasets/kitti_rm_ground                        % KITTI dataset
```

## Get started

### Training
In order to train our model on the Flyingthings3D dataset with 8192 points, run the following:

```bash
$ python train.py --num_points 8192 --batch_size 8 --epochs 120 --use_multi_gpu True
```
for the help on how to use the optional arguments, type:
```bash
$ python train.py --help
```

### Evaluation
In order to evaluate our pretrained model under the ```pretrained_model``` folder with the Flyingthings3D dataset, run the following:

```bash
$ python evaluate.py --num_points 8192 --dataset f3d --ckp_path ./pretrained_model/OGSFNet_94.8932_090_0.1636.pth
```

for the evaluation on KITTI dataset, run the following:
```bash
$ python evaluate.py --num_points 8192 --dataset kitti --ckp_path ./pretrained_model/OGSFNet_94.8932_090_0.1636.pth
```
For help on how to use this script, type:
```bash
$ python evaluate.py --help
```

### Performance
All the following experiments were tested on a single GTX2080Ti GPU

1. Evaluationg results on Flyingthings3D and KITTI datasets:

|                          | EPE_full | EPE    | ACC05  | ACC10  | Outliers |
|--------------------------|----------|--------|--------|--------|----------|
| Flyingthings3D           | 0.1634   | 0.1217 | 0.5518 | 0.7767 | 0.5180   |
| KITTI (without fine tune)| 0.0751   | ~      | 0.7060 | 0.8693 | 0.3277   |

2. Inference time on the test set of Flyingthings3D and KITTI datasets:

| batch size | Flyingthings3D(min) | KITTI(min) |
|------------|----------------|-------|
|      5     |        1.35    | 0.21      |

### Acknowledgement
In this project we use parts of the official implementations of the following libraries and repositories:
* [Pointnet2_PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)
* [PointPWC-Net](https://github.com/DylanWusee/PointPWC)
