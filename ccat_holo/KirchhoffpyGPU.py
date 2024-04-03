#%%
import os
import numpy as np;
import torch as T;
import copy;
import time;
# %%
'''
1. Define Electromagnetic-Field Data, which is a vector and each component is a complex value
'''
class Complex():
    '''
    field is combination of real and imag parts to show the phase informations
    '''
    
    def __init__(self):
        self.real=np.array([]);
        self.imag=np.array([]);
        
    def np2Tensor(self,DEVICE=T.device('cpu')):
        '''DEVICE=T.device('cpu') or T.device('cude:0')'''
        if type(self.real).__module__ == np.__name__:
            self.real=T.tensor(self.real,dtype=T.float64).to(DEVICE)
        elif type(self.real).__module__==T.__name__:
            self.real=self.real.to(DEVICE);            
        if type(self.imag).__module__ == np.__name__:
            self.imag=T.tensor(self.imag,dtype=T.float64).to(DEVICE)
        elif type(self.imag).__module__==T.__name__:
            self.imag=self.imag.to(DEVICE)
        else:
            print('The input data is wrong')
            
    def Tensor2np(self):
        if type(self.real).__module__==T.__name__:
            self.real=self.real.cpu().numpy();
            
        if type(self.imag).__module__==T.__name__:
            self.imag=self.imag.cpu().numpy();
        else:
            pass;
'''1. Define Kirchhoff-Fresnel Integration Solver.'''
def PO_scalar(m1,m1_n,ds,m2,cos_i,Field_m1,k,Keepmatrix=False,Device='cuda',**keywards):
    '''
    Formulatar of the Kirchhoff-Fresnel Integration method
    *******Field=Sum{field()*exp(-j*k*r)/R*[cos_i+cos_r] J*dx*dy}*************
    **************************************************************************
    m1: reflector of source field;
    m2: target reflector;
    Field_in: Field on 'm1';
    cos_i: cosin of reflection angle on m1, in the following document 
    cos_r is cosin of the output ray from the point on 'm1' to target point on 'm2' ;
    Keepmatrix means if the intermediate computing results are saved as a 2D matrix used for speeding up
    calculations.'
    Device: 'cpu' means the computation is done by cpu. 'cuda' can load the data into
    GPU RAM and computed by GPU for acceleration.
    '''
    N=m2.x.size
    cos_i_size=cos_i.size
    m1.np2Tensor(Device)
    m1_n.np2Tensor(Device)
    m2.np2Tensor(Device)
    cos_i=T.tensor(cos_i,dtype=T.float64).to(T.device(Device))
    cos_i = cos_i.to(Device)
    k=T.tensor([k]).to(Device)
    ds=T.tensor(ds).to(Device)
    #=T.tensor(Field_m1,dtype=T.complex128).to(Device)
    F_m1=T.tensor(Field_m1.real,dtype=T.float64)+1j*T.tensor(Field_m1.imag,dtype=T.float64)
    F_m1=F_m1.to(Device)
    if Device=='cuda':
        print('The computation is speed up by GPU units!')            
    # Define output field:
    Field_m2=Complex()
    Field_m2.real=T.zeros(N,dtype=T.float64)
    Field_m2.imag=T.zeros(N,dtype=T.float64)
    Field_m2.np2Tensor(Device)
    
    def calcu(x,y,z):
        x=x.reshape(-1,1)-m1.x.reshape(1,-1)
        y=y.reshape(-1,1)-m1.y.reshape(1,-1)
        z=z.reshape(-1,1)-m1.z.reshape(1,-1)
        r=T.sqrt(x**2+y**2+z**2)
        #cosin terms
        if cos_i_size==1:
            cos=1/r
        else:
            cos=T.abs(x*m1_n.x.reshape(1,-1)
                +y*m1_n.y.reshape(1,-1)
                +z*m1_n.z.reshape(1,-1))
            cos=(cos/r+T.abs(cos_i))/2/r 
        del(x,y,z)
        # field calculation 
        Amp=m1_n.N*ds/2/T.pi*T.abs(k)*cos #
        Mat0=Amp*T.exp(-1j*k*r)
        F_m2=T.sum(Mat0*F_m1.reshape(1,-1),axis=-1)  
        return F_m2
    if Device=='cuda':
        M_all=T.cuda.get_device_properties(0).total_memory
        M_element=m1.x.element_size() * m1.x.nelement()
        cores=int(M_all/M_element/15)
        print('cores:',cores)
    else:
        cores=os.cpu_count()*20

    N=m2.x.nelement()
    Ni=int(N/cores)
    for i in range(Ni):
        F=calcu(m2.x[i*cores:(i+1)*cores],m2.y[i*cores:(i+1)*cores],m2.z[i*cores:(i+1)*cores])
        Field_m2.real[i*cores:(i+1)*cores]=F.real
        Field_m2.imag[i*cores:(i+1)*cores]=F.imag
    
    if int(N%cores)!=0:
        F=calcu(m2.x[Ni*cores:],m2.y[Ni*cores:],m2.z[Ni*cores:])
        Field_m2.real[Ni*cores:]=F.real
        Field_m2.imag[Ni*cores:]=F.imag

    m1.Tensor2np()
    m1_n.Tensor2np()
    m2.Tensor2np()
    Field_m2.Tensor2np()
    T.cuda.empty_cache()
    return 1,Field_m2,1