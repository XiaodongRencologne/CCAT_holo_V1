#!/usr/bin/env python
# coding: utf-8

# In[94]:


import numpy as np;
import torch as T;
import numpy.fft as fft;
#k=1.380649*10**(-23);


def cross_correlating(map1,map2,snr1,snr2,B,t):
    '''
    para1: map1 and map2 are the measured data from signal Rx and reference Rx;
           Both of the beams have the peak value of 1;
    
    para2: T1,T2 are the noise temperature in Rx systems;
    
    para3: dB the bandwidth;
    
    para4: t integration time.
    
    '''
    noise1=10**(-snr1/20)
    noise2=10**(-snr2/20)
    map1=map1.T;
    map1=map1[0,...]+1j*map1[1,...]
    map2=map2.T;
    map2=map2[0,...]+1j*map2[1,...]
    N=int(t*B);
    MAP=np.zeros((map1.size))+1j*np.zeros((map1.size));
    for i in range(map1.size):
        # real parts;
        if snr1!=0:
            n1=np.random.normal(0,noise1,N)+1j*np.random.normal(0,noise1,N); 
        else:
            n1=0.0;
        # imag parts;
        if snr2!=0:
            n2=np.random.normal(0,noise2,N)+1j*np.random.normal(0,noise2,N); 
        else:
            n2=0.0;
        # get the map1 and map2
        s1=map1[i]+n1;
        s2=map2[i]+n2;
        # multiplier 
        data=(s1*np.conjugate(s2)).mean();
        MAP[i]=data;
        #MAP[0,i]=data.real;
        #MAP[1,i]=data.imag;
    
    return MAP;
        

def Random_Gain(dG,Map):
    '''
    dG units is dB;
    '''
    Map=Map.T;
    Map=Map[0,...]+1j*Map[1,...];    
    Gain=1+np.random.normal(0,dG*100,Map.size)/100;
    data=Map*Gain;
    
    return data;


def fluctuation(dG,Map,dt,cut_f):
    '''
    dG units is dB;
    '''
    Map=Map.T;
    Map=Map[0,...]+1j*Map[1,...];
    dG=np.random.normal(0,dG*100,Map.size)/100;
    freq=fft.fftfreq(Map.size,dt);
    dG_f=fft.fft(dG);
    NN=np.where(np.abs(freq)>cut_f);
    dG_f[NN]=0+1j*0;
    dG=fft.ifft(dG_f).real
    Gain=1+dG;
    data=Map*Gain;
    
    return np.append(data.real,data.imag).reshape(2,-1);


def Random_phase(dP,Map):
    '''
    dP units is degree;
    '''
    Map=Map.T;
    Map=Map[0,...]+1j*Map[1,...];
    Phase=np.exp(1j*np.random.normal(0,dP,Map.size)/180*np.pi);
    data=Map*Phase;
    
    return data;






    
    
    

