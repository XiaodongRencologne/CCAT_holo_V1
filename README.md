# 'ccat_holo'

'ccat_holo' is a python package which is designed to analyze the holographic measurement data for the Fred Young Sub-mm Telescope. The goal is to simultaneously measure the FYST's two mirror surfaces. The surface precision of the telescope must be better than 10um (goal of <7um>). This requires that the measurement accuracy of the system is <3um.

The new **'Multi-map'** Holography method has been developed for discriminating the surface errors of the two reflectors of FYST by taking 5 different beam maps. This 'ccat_holo' package was developed for the data analysis which can convert the 5 measured beam maps into 'Two' surface error maps.

## Installation

### This package just work with python3 with verion >=3.7

Following packages are required:
1. numpy v1.21
2. scipy v1.7
3. pytorch 1.12
4. transforms3d v0.4.1
5. h5py v3.6.0
6. pyvista

You can install these packages using the command:  
```shell
'pip install -r requirements' 
``` 
When you have all required packages, you can clone or download the 'ccat_holo' repository from github or uni-koeln gitlab.  
```shell
git clone https://github.com/XiaodongRencologne/CCAT_holo_V1.git
```

## Usage

The details of the software are explained in the [Documentation_ccatholo](Documentation_ccatholo), which includes the presentation of the FYST geometry model and the data analysis procedure. 

The examples of using the software for the FYST analysis is the folder 'examples' including:

1. [Initialization the FYST holo-system](examples/1_initialization_FYST_holo.ipynb)
2. [Data analysis](examples/2_Analysis.ipynb)
3. [Analysis of Large spatial mirror deforamtions](examples/3_Fit_Mirror_Surfaces_Zernike.ipynb)
