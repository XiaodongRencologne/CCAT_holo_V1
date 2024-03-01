# ccat_holo

'ccat_holo' is a python package which is designed to analyze the holographic measurement data for the Fred Young Sub-mm Telescope. The telescope uses a special 'Crossed-Dragone' optics design, which consists of two 6-m reflectors. Its two reflectors both have to be aligned to be better than 10um (goal of < 7um). 

The new **'Multi-map'** Holography method has been developed for measuring and discriminating the surface errors of the two reflectors of FYST by taking 5 different beam maps. The software was developed for the data analysis which can convert the 5 measured beam maps into 'Two' surface error maps.

1. [Installation](#Installation)
2. [FYST Geometry](#FYST-Geometry)
3. [Coordinate Systems](#Coordinate-Systems)
4. [Configurate the FYST Holography](#Configurate-the-FYST-Holography)
    1. [Set electrical parameters: operating frequency and the holo-Rx beam](#1-set-electrical-parameters-operating-frequency-and-the-holo-rx-beam)
    2. [Define the FYST Holo-model](#2-define-and-initialize-the-fyst-holo-model)
    3. [Start the First Time-cost Beam calculation](#3-start-the-first-time-cost-beam-calculation)

## Installation
**This package just works with python3.**

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
or
```shell
git clone https://github.com/XiaodongRencologne/CCAT_holo_V1.git
```

## FYST Geometry
The details of the FYST geometry (Figure 1) is defined by files in the 'CCAT_model' folder. **For the FYST holographic analysis, you don't need to modifiy anything.**

![image info](pictures/FYST_model1.png)  
*Figure 1: FYST optical layout and 5 receiver locations.*

Here, we explain the parameters in the files.  
**Mirror surface profile: 'coeffi_m1.txt/coeffi_m2.txt'**
The files record the coefficients of the 2D polynominals which are used to describe the surface profiles of the FYST's two mirrors (M1 and M2). x and y coordinates of the 2D polynomials are normalized by a factor of 3000mm (radius of the FYST's aperture). The two mirror surfaces are defined in their local coordinate systems.   
$$z=\sum_{i,j=0}{c_{ij}{\left({x}\over R\right)}^{i}{\left({y}\over R\right)}^{j}}$$

**Panel positions: 'L_m1.txt and L_m2.txt':**
The files define the panel layout in M1 and M2. They record the list of panel center positions. 

**Sampling number: 'Model.txt':**
The size of the mirror panels (700x710mm on M2 and 670x750mm on M1) and positions of the 4 corner panel adjusters can be set in this file.  The required sampling points on the mirror panels are defined in the file. The sampling range and points in the intermediate focal (IF) plane are also given here. 
|                    |  Size        |  Sampling Points  |
| ------------------ |--------------| ----------------- |
| M2                 | 700x710mm    | 15x15             |
| M1                 | 670x750mm    | 14x14             |
| IF                 | 540x540mm    | 121x121           |

## Coordinate Systems


FYST holography meausrement will measure 5 beam maps by puting the Rx at 5 different points. The receiver mounting points and the cooresponding antenna scanning trajectory must be expressed in the **'coord_Rx'** & **'coord_Scan'** coordinate systems indicated in below Figure 2.

In the practical holographic measurement, The coordinates of the recorded field points need to be converted into the points expressed in the 'coord_scan' frame.

**'Coord_M1' and 'Coord_M2'** are the frames for defining the mirror layout.

![image info](pictures/FYST_optics.png)

*Figure 2. Optical layout of FYST and its coordinate systems.*


## Configurate the FYST Holography
### 1 Set electrical parameters: operating frequency and the holo-Rx beam


The measuring frequency of the holo-system and Guassian beam of the used holo-Rx are given in file 'electrical_parameter.txt'.
|  Parameters        |  Sampling Points  |
|--------------------| ----------------- |
| freq/GHz           | 296               |
| Edge_taper/dB      | -8                |
| taper_angle/deg    | 11.894            |

The Gaussian beam of the receiver is set by the illumination edge taper at a specific taper angle. 

### 2 Define and initialize the FYST Holo-model

All required methods for the FYST holography data analysis are integrated in the class <em>**'CCAT_holo'**<em> in the package <em>**ccat_holo.Pyccat**<em>. Before starting the data analysis, we should first define the FYST holography model correctly. In the following example, I will demonstrate the code about the holo-mode defination. You also can run the code in jupyter notebook [Initialization_FYST_holo](examples/1_initialization_FYST_holo.ipynb).


```python
# 0. Import the CCAT_holo model
from ccat_holo.Pyccat import CCAT_holo
import torch as T # import pytorch package
```

<em>**CCAT_holo**<em> requires 3 input parameters. First is the folder 'CCAT_model' defining the [FYST geometry model](#fyst-geometry).  
<em>**Output_folder**<em> is the chosen folder to store the matrixes of the initial beam calculation.  
Then we should define 5 receiver mounting points and their measured field points. These are set by a python dictionary seen below.


```python
# 1. Model folder
Model_folder='CCAT_model'
# 2. Configuration of the Holography system
holo_setup={'Rx1':([0,0,600],'scan/51/on-axis.txt'), 
            'Rx2':([400,400,600],'scan/51/400_400_600.txt'),
            'Rx3':([400,-400,600],'scan/51/400_-400_600.txt'),
            'Rx4':([-400,400,600],'scan/51/-400_400_600.txt'),
            'Rx5':([-400,-400,600],'scan/51/-400_-400_600.txt')
            }
# Define 5 receiver positions in the 'Coord_Rx' system, e.g [+/-400,+/-400,600]
# and their field points that are stored in file 'scan/400_400_600.txt'. The field
# points of the measured beam maps are from the read out of the telescope coders.
#####################
# 3. Output_folder is the folder used to store the intermediate data of the first
#    beam calculation and the analysing results.
Output_folder='Analysis1'

# 4. Create the FYST holography Model and check the telescope model and 
# holographic setup in the 3D view.
Model=CCAT_holo(Model_folder,Output_folder,holo_conf=holo_setup)
```

    FYST telescope model has been created!!
    

- Using the method <em>'Model.view'<em> can show the 3D model of the defined antenna and the pre-defined 5 Receiver points. 


```python
Model.view() # Show 3D FYST model and its receivers.
```

![image info](pictures/FYST_3D.png)


```python
Model.view_Rx(Rx=['Rx1']) # This method can highlight the chosen reciever horns.
```

### 3. Start the First Time-cost Beam calculation
This software package uses fitting algorithm and requires thousands fitting loop to find the best fit panel distortions. To speed up the beam calculation called forward function, we run a 'time-cost' and accurate beam calculation, and save the intermediate values (e.g., complex field values on M1, M2, IF Plane and the desired source region, and the field transfer metrixes between M1, M2 and the spherical face in source) as HDF5 binary data format and store the data in the given <em>**Output_folder**<em>. Combining several linear approximations can significantly reduce computing time by three orders of magnitude.

The first time-cost beam calculation can be done by running the function <em>**Model.First_Beam_cal()**<em>.  

**For one independent holographic measurement, we just need to do the time-cost beam calculation once.** 


```python
# 5. Running the time-consuming first beam calculation.
''' We only need to run this calculation in the beginning
 of the data analysis. All the setup defined in 'holo_config'
 will be computed. The intermediate computed data will be
 stored in the directory 'output_folder', here is 'Analysis1'.
'''
Model.First_Beam_cal()


```

    The holographic setup:
    Rx1 : [0, 0, 600] scan/on-axis.txt
    Rx2 : [400, 400, 600] scan/400_400_600.txt
    Rx3 : [400, -400, 600] scan/400_-400_600.txt
    Rx4 : [-400, 400, 600] scan/-400_400_600.txt
    Rx5 : [-400, -400, 600] scan/-400_-400_600.txt
    
    ***Start the initial beam calculations 
    ***and prepare the required Matrixes used to speed up the forward beam calculations.
    Rx1 : [0, 0, 600] scan/on-axis.txt
    time used: 149.97290759999998
    Rx2 : [400, 400, 600] scan/400_400_600.txt
    time used: 151.30981620000003
    Rx3 : [400, -400, 600] scan/400_-400_600.txt
    time used: 155.64836820000005
    Rx4 : [-400, 400, 600] scan/-400_400_600.txt
    time used: 145.1400549
    Rx5 : [-400, -400, 600] scan/-400_-400_600.txt
    time used: 156.17648170000007
    




    " We only need to run this calculation in the beginning\n of the data analysis. All the setup defined in 'holo_config'\n will be computed. The intermediate computed data will be\n stored in the directory 'output_folder', here is 'Analysis1'.\n"



## Make the 'Forward' Beam calculation function (Model.FF)
**Linear approximation**
Because the surface deformations of the two mirrors are much smaller than the operating wavelength (~1mm), we implement linear approximations to speed up the beam computation for the given panel distortions, which assums that the panel distortions only modify the phase of the EM fields on the surfaces of M1 and M2 and the phase changes is linear to the panel offset.


Using the results from the costly first beam computation and the linear approximations, the computing time of one beam prediction can be reduced to <1s. Implementing the advanced GPU technique can further accelerate the calculation. 

<em>**Model.FF**<em> is the created beam forward function with panel adjuster movements as its input parameters.  
<em>**Model.FF(S,Amp_ap, Phase_ap)**<em>: S is the displacement of panel adjusters on M1 and M2, <em>Amp_ap<em> is a list of parameters used to correct the illumination tape of the telescope. And <em>Phase_ap<em> is used to express the field phase difference in aperture between the designed holography model and the practical measurement.

We can build the function <em>**Model.FF**<em> by runing <em>**Model.mk_FF(fitting_param='panel adjusters',Device=T.device('cpu'))**<em>. If <em>Device<em> is set by "T.device('cude:0')" and the Nvidia Graphic card is available, the model can be loaded onto Graphic card for acceleration.




```python
# 6. Make the farward beam calculation function, (Model.FF), with using
# the data produced by the First beam calculations. Here, we can use the movements
# of the panel adjusters or coefficients of Zernike polynomial as the fitting paramters.
# If fitting_param is 'zernike', the surface deviations will be described by 
# the summation of zernike polynimals, and the maximum zernike order is 7th.
Model.mk_FF(fitting_param='panel adjusters',Device=T.device('cpu'))
#Model.mk_FF(fitting_param='panel adjusters',Device=T.device('cuda:0'))

# The function also supports the GPU acceleration by setting the 'Device' to 
# 'T.device('cuda:0')'.

# Model.FF(adjusters, Para_Amp, Para_phase)

```


```python
# 7. Start
```
