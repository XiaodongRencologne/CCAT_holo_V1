#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np;
import matplotlib.pyplot as plt;
'''
1. get the field on M1 and M2
'''
Field_m2=np.genfromtxt('input_beams\in-focus\m2_field.txt',delimiter=',')
Field_m1=np.genfromtxt('input_beams\in-focus\m1_field.txt',delimiter=',')
'''
2. get the panel sampling data
'''
N_m2=15
N_m1=13
Field_m2=(Field_m2[...,0]+1j*Field_m2[...,1]).reshape(-1,N_m2*N_m2);
Field_m1=(Field_m1[...,0]+1j*Field_m1[...,1]).reshape(-1,N_m1*N_m1);
Field_m2=np.sqrt((np.abs(Field_m2)**2).sum(axis=1));
Field_m1=np.sqrt((np.abs(Field_m1)**2).sum(axis=1));
'''
3. calculate the weights of each panel
'''
Weight_m2=Field_m2/Field_m2.max();
Weight_m1=Field_m1/Field_m1.max();

'''
4. calculate the weighted surface error
'''
def Error_rms(Ref_ad,fitting_ad):
    error=np.abs(Ref_ad-fitting_ad);
    error2=error[0:5*69].reshape(5,-1);
    error1=error[5*69:].reshape(5,-1);
    #plt.plot(error2[0,...]*1000,'*--')
    error2=error2*Weight_m2;
    error1=error1*Weight_m1;
    '''
    plt.plot(error2[0,...]*1000,'*--')
    plt.show()
    '''
    rms2=np.sqrt((error2**2).mean())*1000;
    rms1=np.sqrt((error1**2).mean())*1000;
    return rms2,rms1
