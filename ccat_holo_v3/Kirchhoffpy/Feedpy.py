#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
This package provides N input beams, and each beam function can offer scalar and vector modes.
1. Gaussian beam in far field;
2. Gaussian beam near field;
'''

import numpy as np;
from .coordinate_operations import cartesian_to_spherical as cart2spher;
from .coordinate_operations import Transform_local2global as local2global;
from .coordinate_operations import Transform_global2local as global2local;



# In[5]:


'''
Type 1: Gaussian beam;
'''

def Gaussibeam(Edge_taper,Edge_angle,k,Mirror_in,Mirror_n,angle,displacement,polarization='scalar'):
    '''
    param 1: 'Edge_taper' define ratio of maximum power and the edge power in the antenna;
    param 2: 'Edge_angle' is the angular size of the mirror seen from the feed coordinates;
    param 3: 'k' wave number;
    param 4: 'Mirror_in' the sampling points in the mirror illumanited by feed;
    param 5: 'fieldtype' chose the scalar mode or vector input field.
    '''
    
    Mirror_in=global2local(angle,displacement,Mirror_in);
    Mirror_n=global2local(angle,[0,0,0],Mirror_n);
    if polarization.lower()=='scalar':
        '''
        Edge_taper=np.abs(Edge_taper);
        waist2=20*np.log10(np.exp(1))*Edge_angle**2/Edge_taper;
        r,theta,phi=cart2spher(Mirror_in.x,Mirror_in.y,Mirror_in.z);        
        Amp=1000*np.exp(-theta**2/waist2)/r;
        Phase0=-k*r;
        
        Field_R=Amp*np.cos(Phase0);
        Field_I=Amp*np.sin(Phase0);
        cos_i=np.abs(Mirror_in.x*Mirror_n.x+Mirror_in.y*Mirror_n.y+Mirror_in.z*Mirror_n.z)/r;
        return Field_R,Field_I,cos_i;
        '''
        Theta_max=Edge_angle;
        E_taper=Edge_taper;
        b=(20*np.log10((1+np.cos(Theta_max))/2)-E_taper)/(20*k*(1-np.cos(Theta_max))*np.log10(np.exp(1)));
        w0=np.sqrt(2/k*b)
        r,theta,phi=cart2spher(Mirror_in.x,Mirror_in.y,Mirror_in.z);
        R=np.sqrt(r**2-b**2+1j*2*b*Mirror_in.z);
        E=np.exp(-1j*k*R-k*b)/R*(1+np.cos(theta))/2/k/w0*b;
        E=E*np.sqrt(8);
                
        cos_i=np.abs(Mirror_in.x*Mirror_n.x+Mirror_in.y*Mirror_n.y+Mirror_in.z*Mirror_n.z)/r;

        return E.real,E.imag,cos_i;
    
  
    
    

