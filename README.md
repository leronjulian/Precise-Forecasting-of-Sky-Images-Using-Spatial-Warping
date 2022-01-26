# Precise Forecasting of Sky Images Using Spatial Warping
 SkyNet imrpoves sky-image prediction to model cloud dynamics with higher spatial and temporal resolution than previous works. Our method handles distorted clouds near the horizon of the hemispherical mirror by patially warping the sky images during training to facilitate longer forecasting of cloud evolution. 

# To download dataset for train and test data:

- `pip install gdown`
- `gdown --folder --id 1BkWx0j6Kt5G8CEMzzREprMeoYfw0v4ge`
    
# Installation

Installation using using anaconda package management

- `conda env create -f environment.yml`

- `conda activate SkyNet`

- `pip install -r requirements.txt`

# How to train the model with default parameters:
    python train.py

# For info about command-line flags use
    python train.py --help

# Running Tests (WORK-IN-PROGRESS)
    python test.py

# Citation
```
[1]  @misc{pytorch-liteflownet,
         author = {Simon Niklaus},
         title = {A Reimplementation of {LiteFlowNet} Using {PyTorch}},
         year = {2019},
         howpublished = {\url{https://github.com/sniklaus/pytorch-liteflownet}}
    }

[2]  @inproceedings{Hui_CVPR_2018,
         author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},
         title = {{LiteFlowNet}: A Lightweight Convolutional Neural Network for Optical Flow Estimation},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2018}
     }
[3]  @inproceedings{ICCV_2021,
         author = {Leron Julian},
         title = {Precise Forecasting of Sky Images Using Spatial Warping},
         booktitle = {In Proceedings of the IEEE/CVF International Conference on Computer Vision},
         year = {2-21}
     }
```


