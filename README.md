# Precise Forecasting of Sky Images Using Spatial Warping
 SkyNet imrpoves sky-image prediction to model cloud dynamics with higher spatial and temporal resolution than previous works. Our method handles distorted clouds near the horizon of the hemispherical mirror by patially warping the sky images during training to facilitate longer forecasting of cloud evolution. 

# To download dataset for train and test data:
    
[ https://drive.google.com/drive/folders/1BkWx0j6Kt5G8CEMzzREprMeoYfw0v4ge?usp=sharing ]
# Installation
Installation using using anaconda package management

'conda env create -f environment.yml'
'conda activate SkyNet'

# How to train the model with default parameters:
    python train.py

# For info about command-line flags use
    python train.py --help

# Running Tests
    python test.py



