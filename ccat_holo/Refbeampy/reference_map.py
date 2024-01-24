#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np;
import torch as T;

from .coordinate_operations import cartesian_to_spherical as cart2spher;
from .coordinate_operations import Transform_local2global as local2global;
from .coordinate_operations import Transform_global2local as global2local;


# In[3]:


def gaussianbeam(x,y,z,BW,k,fieldtype='far'):
    if fieldtype=='far':
        theta_max=2/k/BW;
        Amp=np.exp(-(np.arctan(np.sqrt(x**2+y**2)/z)**2)/theta_max**2);
        phase=0;
        Field=Amp*np.exp(1j*phase);
            
    return Field.real,Field.imag;
        
    
        
    

