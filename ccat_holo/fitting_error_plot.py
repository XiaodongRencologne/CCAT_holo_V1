#!/usr/bin/env python
# coding: utf-8

# In[5]:
import sys,os
sys.path.append('.')
import numpy as np;
import matplotlib.pyplot as plt;
from mirrorpy import deformation,adjuster,model_ccat

from zernike_torch import mkCFn as make_zernike;

c=299792458

def read_input(inputfile):
    coefficient_m2=np.genfromtxt(inputfile+'/coeffi_m2.txt',delimiter=',');
    coefficient_m1=np.genfromtxt(inputfile+'/coeffi_m1.txt',delimiter=',');
    List_m2=np.genfromtxt(inputfile+'/L_m2.txt',delimiter=',');
    List_m1=np.genfromtxt(inputfile+'/L_m1.txt',delimiter=',');
    parameters=np.genfromtxt(inputfile+'/model.txt',delimiter=',')[...,1];
    electro_params=np.genfromtxt(inputfile+'/electrical_parameters.txt',delimiter=',')[...,1];
    
    M2_size=parameters[0:2];M1_size=parameters[2:4];
    R2=parameters[4];R1=parameters[5];
    p_m2=parameters[6];q_m2=parameters[7];
    p_m1=parameters[8];q_m1=parameters[9];
    M2_N=parameters[10:12];M1_N=parameters[12:14]
    fimag_N=parameters[14:16];fimag_size=parameters[16:18]

    freq=electro_params[0]*10**9;
    edge_taper=electro_params[1];
    Angle_taper=electro_params[2]/180*np.pi;
    Lambda=c/freq*1000;
    k=2*np.pi/Lambda;
    
    return coefficient_m2,coefficient_m1,List_m2,List_m1,M2_size,M1_size,R2,R1,p_m2,q_m2,p_m1,q_m1,M2_N,M1_N,fimag_N,fimag_size,edge_taper,Angle_taper,k
    




'''
define function to reshape the mirror to mesh grid points;
'''
def reshape_model(sizex,sizey,m,z,dy=1,Num=11):
    x0=np.linspace(-4.5*sizex+sizex/Num/2,4.5*sizex-sizex/Num/2,Num*9)
    y0=np.linspace(-4.5*sizey+sizey/Num/2,4.5*sizey-sizey/Num/2,Num*9)
    #m2
    y0=y0+dy
    #m1
    #y0=y0;
    x,y=np.meshgrid(x0,y0)
    dz=np.zeros(x.shape)
    for i in range(9*Num):
        for n in range(9*Num):        
            a=np.where((m.x>(x[i,n]-0.001))&(m.x<(x[i,n]+0.001)) &(m.y>(y[i,n]-0.001))&(m.y<(y[i,n]+0.001)))        
            if a[0].size:
                dz[i,n]=z[a];
            else:
                dz[i,n]=np.nan;
    return x,y,dz


# In[18]:


'''
define a color map plot function;
'''
def colormap(x1,y1,z1,x2,y2,z2,Vmax=None,Vmin=None,suptitle=''):
    cmap = plt.get_cmap('hot');
    font = {'family': 'serif','color':'darkred','weight':'normal','size':16};
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(14,6));
    ax1=ax[0];
    ax2=ax[1];
    p1=ax1.pcolor(x2,y2,z2,cmap=cmap,vmin=Vmin,vmax=Vmax);
    ax1.axis('scaled');
    ax1.set_xlabel('Secondary mirror',fontdict=font);
    clb=fig.colorbar(p1, ax=ax1,shrink=0.95,fraction=.05);
    clb.set_label('um',labelpad=-40,y=1.05,rotation=0)
    p1=ax2.pcolor(x1,y1,z1,cmap=cmap,vmin=Vmin,vmax=Vmax);
    ax2.axis('scaled');
    ax2.set_xlabel('Primary mirror',fontdict=font);
    clb=fig.colorbar(p1, ax=ax2,shrink=0.95,fraction=.05);
    clb.set_label('um',labelpad=-40,y=1.05,rotation=0);
    fig.suptitle(suptitle,fontsize=15,color='k',verticalalignment='top')#'baseline')
    plt.show()


# In[19]:
'''
Draw the panel errors on M1 and M2
'''
def Fit_M_Surface(Fit_S,vmax=100,vmin=-100,
                  Ref_S=None,Ref_vmax=100,Ref_vmin=-100,
                  diff_rms=10,model_file='CCAT_model'):
    """
    'Fit_S' is the solution of the fitted panel adjuster errors.
    'Ref_S' is the values of the panel adjusters. If this parameter is 'None, only the fitted solution will be plotted.
    """
    '''
    read input data file
    '''
    
    coefficient_m2,coefficient_m1,List_m2,List_m1,M2_size,M1_size,R2,R1,p_m2,q_m2,p_m1,q_m1,M2_N,M1_N,fimag_N,fimag_size,edge_taper,Angle_taper,k=read_input(model_file)

    M2_size=M2_size+1.2;
    M1_size=M1_size+1.2;
    M2_N=[13,13]
    M1_N=[13,13]
    '''
    build model
    '''
    ad_m2=np.zeros(5*List_m2.shape[0]);
    ad_m1=np.zeros(5*List_m1.shape[0]);
    ad=np.append(ad_m2,ad_m1).ravel()
    m2,m2_n,m2_dA,m1,m1_n,m1_dA,fimag,fimag_n,fimag_dA=model_ccat(coefficient_m2,List_m2,M2_size[0],M2_size[1],M2_N[0],M2_N[1],R2,
                                                                    coefficient_m1,List_m1,M1_size[0],M1_size[1],M1_N[0],M2_N[1],R1,
                                                                    fimag_size[0],fimag_size[1],fimag_N[0],fimag_N[1],
                                                                    ad,p_m2,q_m2,p_m1,q_m1)

    dx2,dy2,dx1,dy1=adjuster(List_m2,List_m1,p_m2,q_m2,p_m1,q_m1,R2,R1)
    #1. panel error from holography fitting
    ad_m2=Fit_S[0:5*List_m2.shape[0]]
    ad_m1=Fit_S[5*List_m2.shape[0]:5*(List_m2.shape[0]+List_m1.shape[0])]
    ''' reshape the error matrixes shape'''    
    M20=deformation(ad_m2.ravel(),List_m2,p_m2,q_m2,m2)
    M10=deformation(ad_m1.ravel(),List_m1,p_m1,q_m1,m1)
    x2,y2,dz2_1=reshape_model(M2_size[0],M2_size[1],m2,M20,dy=-1,Num=int(M2_N[0]))
    x1,y1,dz1_1=reshape_model(M1_size[0],M1_size[1],m1,M10,dy=35,Num=int(M1_N[0]))

    '''
    1. Solutions, fitted panel errors.
    '''
    colormap(x1,y1,dz1_1*1000,
             x2,y2,dz2_1*1000,
             Vmax=vmax,Vmin=vmin,
             suptitle='Fitting Mirror Maps');
    
    if Ref_S is None:
        pass
    else:
        ad_m2=Ref_S[0:5*List_m2.shape[0]]
        ad_m1=Ref_S[5*List_m2.shape[0]:]
        ''' reshape the error matrixes shape'''
        M20=deformation(ad_m2.ravel(),List_m2,p_m2,q_m2,m2)
        M10=deformation(ad_m1.ravel(),List_m1,p_m1,q_m1,m1)
        x2,y2,dz2_0=reshape_model(M2_size[0],M2_size[1],m2,M20,dy=-1,Num=int(M2_N[0]))
        x1,y1,dz1_0=reshape_model(M1_size[0],M1_size[1],m1,M10,dy=35,Num=int(M1_N[0]))

        '''
        2. Reference panel errors.
        '''
        colormap(x1,y1,dz1_0*1000,
                 x2,y2,dz2_0*1000,
                 Vmax=Ref_vmax,Vmin=Ref_vmin,
                 suptitle='Reference Mirror Maps')
        
         # 3. calculate the error;    
        err2=(dz2_1-dz2_0)*1000
        err1=(dz1_1-dz1_0)*1000
        rms2=np.sqrt(np.nanmean(err2**2))
        rms1=np.sqrt(np.nanmean(err1**2))

        '''
        2. Reference panel errors.
        '''
        colormap(x1,y1,err1,
                 x2,y2,err2,
                 Vmax=diff_rms,Vmin=-diff_rms,
                 suptitle='Difference')


### plot the zernike results
def Fit_M_Surface_zk(coeff_zk,Z_order,model_file='CCAT_model',vmax=100,vmin=-100):
    coeff_zk=coeff_zk.reshape(2,-1)
    coeff_m2=np.append(np.zeros(3),coeff_zk[0,:])
    coeff_m1=np.append(np.zeros(3),coeff_zk[1,:])

    coefficient_m2,coefficient_m1,List_m2,List_m1,M2_size,M1_size,R2,R1,p_m2,q_m2,p_m1,q_m1,M2_N,M1_N,fimag_N,fimag_size,edge_taper,Angle_taper,k=read_input(model_file)

    M2_size=M2_size+1.2;
    M1_size=M1_size+1.2;
    M2_N=[13,13]
    M1_N=[13,13]
    '''
    build model
    '''
    ad_m2=np.zeros(5*List_m2.shape[0]);
    ad_m1=np.zeros(5*List_m1.shape[0]);
    ad=np.append(ad_m2,ad_m1).ravel()
    m2,m2_n,m2_dA,m1,m1_n,m1_dA,fimag,fimag_n,fimag_dA=model_ccat(coefficient_m2,List_m2,M2_size[0],M2_size[1],M2_N[0],M2_N[1],R2,
                                                                    coefficient_m1,List_m1,M1_size[0],M1_size[1],M1_N[0],M2_N[1],R1,
                                                                    fimag_size[0],fimag_size[1],fimag_N[0],fimag_N[1],
                                                                    ad,p_m2,q_m2,p_m1,q_m1)

    
    #1. panel error from holography fitting
    Z_surf2=make_zernike(Z_order,m2.x/R2,(m2.y+0.0)/R2,dtype='numpy')
    Z_surf1=make_zernike(Z_order,m1.x/R1,(m1.y-0.0)/R1,dtype='numpy')
    
    z2=Z_surf2(coeff_m2)
    z1=Z_surf1(coeff_m1)
    x2,y2,dz2_1=reshape_model(M2_size[0],M2_size[1],m2,z2,dy=-1,Num=int(M2_N[0]))
    x1,y1,dz1_1=reshape_model(M1_size[0],M1_size[1],m1,z1,dy=35,Num=int(M1_N[0]))

    colormap(x1,y1,dz1_1*1000,
             x2,y2,dz2_1*1000,
             Vmax=vmax,Vmin=vmin,
             suptitle='Fitting Mirror Maps');