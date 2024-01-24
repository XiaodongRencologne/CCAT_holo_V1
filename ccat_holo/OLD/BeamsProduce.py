#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Kirchhoffpy.BeamPattern import squarePattern


# In[2]:


u0=0.027;
v0=0.027;
urange=0.01;
vrange=0.01;
Nu=101;
Nv=101;
squarePattern(u0,v0,urange,vrange,Nu,Nv,file='beam',distance=300000,Type='on-axis')
squarePattern(u0,v0,urange,vrange,Nu,Nv,file='beam',distance=300000,Type='off-axis')

