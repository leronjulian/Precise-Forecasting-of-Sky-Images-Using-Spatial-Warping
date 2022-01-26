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
[2]  @misc{pytorch-liteflownet,
         author = {Simon Niklaus},
         title = {A Reimplementation of {LiteFlowNet} Using {PyTorch}},
         year = {2019},
         howpublished = {\url{https://github.com/sniklaus/pytorch-liteflownet}}
    }



