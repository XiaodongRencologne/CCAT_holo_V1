#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np;
import torch as T;
'''
1. produce the random noise (SNR)
'''
def noise(data,snr,DEVICE=T.device('cpu')):
    N=data[0,...].size()[0];
    peak=1;
    noise=peak*10**(-snr/20);
    noise=np.concatenate((np.random.normal(0,noise,N),np.random.normal(0,noise,N),
                          np.random.normal(0,noise,N),np.random.normal(0,noise,N),
                          np.random.normal(0,noise,N),np.random.normal(0,noise,N),
                          np.random.normal(0,noise,N),np.random.normal(0,noise,N))).reshape(8,-1);
    DATA=data+T.tensor(noise).to(DEVICE);
    return DATA;


'''
2. random gain
'''
def RandomGain(data,dG,DEVICE=T.device('cpu')):
    N=data[0,...].size()[0];
    gain1=1+np.random.normal(0,dG*100,N)/100;
    gain2=1+np.random.normal(0,dG*100,N)/100;
    gain3=1+np.random.normal(0,dG*100,N)/100;
    gain4=1+np.random.normal(0,dG*100,N)/100;
    
    Gain=np.concatenate((gain1,gain1,
                          gain2,gain2,
                          gain3,gain3,
                          gain4,gain4)).reshape(8,-1);
    DATA=data*T.tensor(Gain).to(DEVICE);
    return DATA;

'''
3. random phase
'''
def RandomPhase(data,dp,DEVICE=T.device('cpu')):
    '''
    dp in degree
    '''
    N=data[0,...].size()[0];
    Amp1=np.exp(1j*np.random.normal(0,dp*100,N)/100/180*np.pi);
    Amp2=np.exp(1j*np.random.normal(0,dp*100,N)/100/180*np.pi);
    Amp3=np.exp(1j*np.random.normal(0,dp*100,N)/100/180*np.pi);
    Amp4=np.exp(1j*np.random.normal(0,dp*100,N)/100/180*np.pi);
    Amp=np.concatenate((Amp1,Amp2,Amp3,Amp4)).reshape(4,-1);
    del(Amp1,Amp2,Amp3,Amp4);
    
    DATA=np.zeros((4,N))+1j*np.zeros((4,N));
    for n in range(4):
        DATA[n,...]=data[2*n,...].cpu().numpy()+1j*data[2*n+1,...].cpu().numpy();
        DATA[n,...]*=Amp[n,...];
    Data=np.array([]);
    for n in range(4):
        Data=np.append(Data,DATA[n,...].real);
        Data=np.append(Data,DATA[n,...].imag);
    del(DATA);
    Data=T.tensor(Data.reshape(8,-1)).to(DEVICE);
    return Data;
'''
4. Ref_Rx position error
'''
def Ref_Rx_error(data,dy,dz,freq=296,beamfile='beam/',DEVICE=T.device('cpu')):
    '''
    dy effect the results
    '''
    c=299792458;
    k=2*np.pi*freq*10**9/c/1000
    N=data[0,...].size()[0];
    dx=0;
    source=np.genfromtxt(beamfile+'center.txt');
    x=source[...,0];y=source[...,1];z=source[...,2];
    r=np.sqrt(x**2+y**2+z**2)
    dr=np.sqrt((x-dx)**2+(y-dy)**2+(z-dz)**2)-r;
    dp=-k*dr;
    Amp=np.exp(1j*dp);
    DATA=np.zeros((4,N))+1j*np.zeros((4,N));
    for n in range(4):
        DATA[n,...]=data[2*n,...].cpu().numpy()+1j*data[2*n+1,...].cpu().numpy();
        DATA[n,...]*=Amp;
           
    Data=np.array([]);
    for n in range(4):
        Data=np.append(Data,DATA[n,...].real);
        Data=np.append(Data,DATA[n,...].imag);
    Data=T.tensor(Data.reshape(8,-1)).to(DEVICE);
    return Data;


    

