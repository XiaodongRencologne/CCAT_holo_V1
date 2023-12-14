#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np;
import torch as T;
import copy ;
import time;

DEVICE0=T.device('cpu')
c=299792458;
'''
some germetrical parametrs
'''
Theta_0=0.927295218001612; # offset angle of MR;
Ls      = 780.0470675116178;           # distance between focal point and SR
Lm      = 450.0;            # distance between MR and SR;
L_fimag=585.1235655326061*2+Ls;
F=1300;


# In[2]:

# import the Firchhoffpy package;
import Kirchhoffpy;
# the intergration funciton by using scalar diffraction theory;
from Kirchhoffpy.Kirchhoff import Complex,PO_scalar;
# 1. define the guassian beam of the input feed;
from Kirchhoffpy.Feedpy import Gaussibeam_Feed;
# 2. translation between coordinates system;
from Kirchhoffpy.coordinate_operations import Coord;
from Kirchhoffpy.coordinate_operations import Transform_local2global as local2global;
from Kirchhoffpy.coordinate_operations import Transform_global2local as global2local;
from Kirchhoffpy.coordinate_operations import cartesian_to_spherical as cart2spher;
# 3. mirror
from Kirchhoffpy.mirrorpy import profile,squarepanel,deformation,ImagPlane;
# 4. field in source region;
from Kirchhoffpy.Spheical_field import spheical_grid;
# 5. inference function;
from Kirchhoffpy.inference import DATA2CUDA,fitting_func_zernik_Feed,fitting_func_zernik_Feed2;
# 6. zernike function
from Kirchhoffpy.zernike_torch import mkCFn as make_zernike;
from Kirchhoffpy.zernike_torch import N as poly_N;
'''
1. read the input parameters
'''
def read_input(inputfile):
    coefficient_m2=np.genfromtxt(inputfile+'/Lab_coeff_m2_R_400.txt',delimiter=',');
    coefficient_m1=np.genfromtxt(inputfile+'/Lab_coeff_m1_R_400.txt',delimiter=',');
    List_m2=np.genfromtxt(inputfile+'/List_m2.txt',delimiter=',');
    List_m1=np.genfromtxt(inputfile+'/List_m1.txt',delimiter=',');
    parameters=np.genfromtxt(inputfile+'/input.txt',delimiter=',')[...,1];
    electro_params=np.genfromtxt(inputfile+'/electrical_parameters.txt',delimiter=',')[...,1];
    
    M2_size=parameters[0:2];M1_size=parameters[2:4];
    R2=parameters[4];R1=parameters[5];
    p_m2=parameters[6];q_m2=parameters[7];
    p_m1=parameters[8];q_m1=parameters[9];
    M2_N=parameters[10:12];M1_N=parameters[12:14]
    fimag_N=parameters[14:16];fimag_size=parameters[16:18]
    distance=parameters[18];
    freq=electro_params[0]*10**9;
    edge_taper=electro_params[1];
    Angle_taper=electro_params[2]/180*np.pi;
    Lambda=c/freq*1000;
    k=2*np.pi/Lambda;
    
    return coefficient_m2,coefficient_m1,List_m2,List_m1,M2_size,M1_size,R2,R1,p_m2,q_m2,p_m1,q_m1,M2_N,M1_N,fimag_N,fimag_size,distance,edge_taper,Angle_taper,k;
    

'''
2. produce the coordinates relationship;
'''
def relation_coorsys(Theta_0,Ls,Lm,L_fimag,F,defocus,dAngle_fz=0):
    '''
    germetrical parametrs
    
    Theta_0=0.927295218001612; # offset angle of MR;
    Ls      = 12000.0;           # distance between focal point and SR
    Lm      = 6000.0;            # distance between MR and SR;
    L_fimag=18000+Ls;
    F=20000;
    #defocus# is the defocus of receiver;
    '''
    
    '''
    #angle# is angle change of local coordinates and global coordinates;
    #D#     is the distance between origin of local coord and global coord in global coordinates;
    '''

    angle_m2=[-(np.pi/2+Theta_0)/2,0,0] #  1. m2 and global co-ordinates
    D_m2=[0,-Lm*np.sin(Theta_0),0]
    
    angle_m1=[-Theta_0/2,0,0]          #  2. m1 and global co-ordinates
    D_m1=[0,0,Lm*np.cos(Theta_0)]
    
    angle_s=[0,np.pi,0];               #  3. source and global co-ordinates
    D_s=[0,0,0];
    
    angle_fimag=[-Theta_0,0,0];        #  4. fimag and global co-ordinates
    defocus_fimag=[0,0,0];
    defocus_fimag[2]=1/(1/F-1/(Ls+defocus[2]))+L_fimag;
    defocus_fimag[1]=(F+L_fimag-defocus_fimag[2])/F*defocus[1];
    defocus_fimag[0]=(F+L_fimag-defocus_fimag[2])/F*defocus[0];
    D_fimag=[0,0,0]
    D_fimag[0]=defocus_fimag[0];
    D_fimag[1]=defocus_fimag[1]*np.cos(Theta_0)-np.sin(Theta_0)*(L_fimag-defocus_fimag[2]+Lm);
    D_fimag[2]=-defocus_fimag[1]*np.sin(Theta_0)-np.cos(Theta_0)*(L_fimag-defocus_fimag[2]);
    
    # 5. feed and global co-ordinate
    '''
    C=1/(1/Lm-1/F)+defocus[2]+Ls;
    C=21000;
    angle_f=[np.pi/2-defocus[1]/C,0,-defocus[0]/C]; 
    D_f=[defocus[0],Ls+defocus[2]-Lm*np.sin(Theta_0),-defocus[1]];
    '''
    angle_f=[np.pi/2,0,dAngle_fz];    
    D_f=[defocus[0],Ls+defocus[2]-Lm*np.sin(Theta_0),-defocus[1]]
    
    
    return angle_m2,D_m2,angle_m1,D_m1,angle_fimag,D_fimag,angle_f,D_f,angle_s,D_s;

'''
3.  build the CCAT-P model
'''
# build the model for ccat-prime and imaginary plane;
def model_ccat(coefficient_m2,List_m2,M2_sizex,M2_sizey,M2_Nx,M2_Ny,R2,# m2
          coefficient_m1,List_m1,M1_sizex,M1_sizey,M1_Nx,M1_Ny,R1, # m1
          Rangex,Rangey,fimag_Nx,fimag_Ny,# imaginary field
          ad_m2,ad_m1,p_m2,q_m2,p_m1,q_m1):              #fimag & initial position of adjusters;
    
    surface_m2=profile(coefficient_m2,R2);# define the surface function of m2;
    surface_m1=profile(coefficient_m1,R1);# define the surface function of m1;    
    m2,m2_n,m2_dA=squarepanel(List_m2[...,0],List_m2[...,1],M2_sizex,M2_sizey,M2_Nx,M2_Ny,surface_m2);
    m1,m1_n,m1_dA=squarepanel(List_m1[...,0],List_m1[...,1],M1_sizex,M1_sizey,M1_Nx,M1_Ny,surface_m1);
    fimag,fimag_n,fimag_dA=ImagPlane(Rangex,Rangey,fimag_Nx,fimag_Ny);
    
    # modified the panel based on the initial adjusters distribution;
    Panel_N_m2=int(List_m2.size/2)
    
    m2_dz=deformation(ad_m2.ravel(),List_m2,p_m2,q_m2,m2);
    m1_dz=deformation(ad_m1.ravel(),List_m1,p_m1,q_m1,m1);
    
    m2.z=m2.z+m2_dz;
    m1.z=m1.z-m1_dz;
    
    # define the rim2;
    bx=223.10509202155814;
    ay=235.17341612319572;
    y0=7.777107340872568;
    NN=np.where((m2.x**2/bx**2+(m2.y+y0)**2/ay**2)>1)
    #m2_n.N[NN]=0.0;
    m2.z[NN]=10;m2_n.N[NN]=1;m2_n.x[NN]=0.0;m2_n.y[NN]=0.0;m2_n.z[NN]=-1;
    # define the rim1:
    bx=207.201605128906;
    ay=231.65843705765147;
    y0=-2.5000000000010663;
    NN=np.where((m1.x**2/bx**2+(m1.y+y0)**2/ay**2)>1)
    #m1_n.N[NN]=0.0;
    m1.z[NN]=-5;m1_n.N[NN]=1;m1_n.x[NN]=0.0;m1_n.y[NN]=0.0;m1_n.z[NN]=-1;

    return m2,m2_n,m2_dA,m1,m1_n,m1_dA,fimag,fimag_n,fimag_dA;     



'''
4. the function gives the calculation orders;
'''
def First_computing(m2,m2_n,m2_dA, # Mirror 2,
                    m1,m1_n,m1_dA,# Mirror 1,
                    fimag,fimag_n,fimag_dA,defocus, # imaginary focal plane,
                    source,   # source
                    k,Theta_max,E_taper,feed_rotation=0,zernike_order=0,param=np.zeros(1),theta_max=20,Keepmatrix=False): # frequency and edge taper;
    
    start=time.perf_counter();

    angle_m2,D_m2,angle_m1,D_m1,angle_fimag,D_fimag,angle_f,D_f,angle_s,D_s=relation_coorsys(Theta_0,Ls,Lm,L_fimag,F,defocus,dAngle_fz=feed_rotation);
    '''
    1. get the field on m2;
    '''
    
    # get the field on m2 and incident angle in feed coordinates;
    m2=local2global(angle_m2,D_m2,m2);
    m2_n=local2global(angle_m2,[0,0,0],m2_n);
    Field_m2=Complex(); # return 2

    Field_m2.real,Field_m2.imag,cosm2_i,theta_m2,phi_m2=Gaussibeam_Feed(E_taper,Theta_max,k,m2,m2_n,angle_f,D_f);
    
    
    if zernike_order!=0:
        theta_max_x=theta_max/180*np.pi;
        theta_max_y=(theta_max+0)/180*np.pi;
        errorFn=make_zernike(zernike_order,theta_m2*np.cos(phi_m2)/theta_max_x,theta_m2*np.sin(phi_m2)/theta_max_y);
        errorPhase=errorFn(param);
        real=Field_m2.real*np.cos(errorPhase)-Field_m2.imag*np.sin(errorPhase);
        imag=Field_m2.real*np.sin(errorPhase)+Field_m2.imag*np.cos(errorPhase);
        Field_m2.real=real;
        del(real);
        Field_m2.imag=imag;
        del(imag,errorPhase,errorFn);
    else:
        pass;
    
    
    '''
    2. calculate the field on imaginary focal plane;
    '''

    fimag=local2global(angle_fimag,D_fimag,fimag);
    fimag_n=local2global(angle_fimag,[0,0,0],fimag_n);

    Matrix1,Field_fimag,cosm2_r=PO_scalar(m2,m2_n,m2_dA,fimag,cosm2_i,Field_m2,-k,Keepmatrix=Keepmatrix)
    
    
    '''
    3. calculate the field on m1;
    '''
    #print('3')
    m1=local2global(angle_m1,D_m1,m1);
    m1_n=local2global(angle_m1,[0,0,0],m1_n);
    aperture=T.cat((T.tensor(m1.x),T.tensor(m1.y))).reshape(2,-1)
    NN=int(fimag.x.size/2);
    Fimag0=[fimag.x[NN],fimag.y[NN],fimag.z[NN]];
    x=Fimag0[0].item()-m1.x.reshape(1,-1)
    y=Fimag0[1].item()-m1.y.reshape(1,-1)
    z=Fimag0[2].item()-m1.z.reshape(1,-1)
    
    r=np.sqrt(x**2+y**2+z**2);
    cosm1_i=(x*m1_n.x+y*m1_n.y+z*m1_n.z)/r;
    #cosm1_i=T.tensor(cosm1_i).to(DEVICE);
    del(x,y,z,r);
    
    Matrix2,Field_m1,cosm=PO_scalar(fimag,fimag_n,fimag_dA,m1,np.array([1]),Field_fimag,k,Keepmatrix=Keepmatrix)
    del(cosm)
    
    '''
    4. calculate the field in source;
    '''
    #print('4')
    source=local2global(angle_s,D_s,source);
    Matrix3,Field_s,cosm1_r=PO_scalar(m1,m1_n,m1_dA,source,cosm1_i,Field_m1,k,Keepmatrix=Keepmatrix);
    
    '''
    emerging m1 and m2 to m12;
    '''    
    Matrix21=Complex();
    if Keepmatrix:
        Matrix21.real=np.matmul(Matrix2.real,Matrix1.real)-np.matmul(Matrix2.imag,Matrix1.imag);
        Matrix21.imag=np.matmul(Matrix2.real,Matrix1.imag)+np.matmul(Matrix2.imag,Matrix1.real);
    else:
        pass;
   
    elapsed =(time.perf_counter()-start);
    print('time used:',elapsed);
    
    if Keepmatrix:
        return Matrix21,Matrix3,cosm2_i,cosm2_r,cosm1_i,cosm1_r,Field_s,Field_fimag,Field_m1,Field_m2,aperture,theta_m2,phi_m2;
    else:
        return Matrix21,Matrix3,cosm2_i,cosm2_r,cosm1_i,cosm1_r,Field_s,Field_fimag,Field_m1,Field_m2,aperture;


    
'''
5.3 the function is the used to calculate the field
'''
def field_calculation_zernike_Feed(inputfile,source_field,defocus,ad_m2,ad_m1,Para_f,zernike_order,feed_rotation,theta_max=20):
    
    
    
    # 0. read the input parameters from the input files;
    coefficient_m2,coefficient_m1,List_m2,List_m1,M2_size,M1_size,R2,R1,p_m2,q_m2,p_m1,q_m1,M2_N,M1_N,fimag_N,fimag_size,distance,edge_taper,Angle_taper,k=read_input(inputfile);
    
    # 1. produce the coordinate system;
    # 2. build model;
    m2,m2_n,m2_dA,m1,m1_n,m1_dA,fimag,fimag_n,fimag_dA=model_ccat(coefficient_m2,List_m2,M2_size[0],M2_size[1],M2_N[0],M2_N[1],R2,
                                                                  coefficient_m1,List_m1,M1_size[0],M1_size[1],M1_N[0],M1_N[1],R1,
                                                                  fimag_size[0],fimag_size[1],fimag_N[0],fimag_N[1],
                                                                  ad_m2,ad_m1,p_m2,q_m2,p_m1,q_m1);

    # 3.calculate the source beam
    Matrix21,Matrix3,cosm2_i,cosm2_r,cosm1_i,cosm1_r,Field_s,Field_fimag,Field_m1,Field_m2,aperture=First_computing(m2,m2_n,m2_dA,
                                                                                                                    m1,m1_n,m1_dA,
                                                                                                                    fimag,fimag_n,
                                                                                                                    fimag_dA,
                                                                                                                    defocus,
                                                                                                                    source_field,
                                                                                                                    k,Angle_taper,
                                                                                                                    edge_taper,
                                                                                                                    feed_rotation=feed_rotation,                                                                                                                    zernike_order=zernike_order,
                                                                                                                    param=Para_f,                                                                                                                    theta_max=theta_max,
                                                                                                                    Keepmatrix=False);
    
    return Field_s,Field_fimag,Field_m1,Field_m2;

    

'''
7.2. function used to do the first precise calculation and also produce a new simple and more efficient forward function. this function is used to get the large-scale error caused by phase pattern of the feed horn.
'''    
def Make_fitfuc_zernike_Feed(inputfile,sourcefile,ad_m2,ad_m1,defocus0,Zernike_order,dA_f0,dA_f1,dA_f2,theta_max=20):
    
    # 0. read the input parameters from the input files;
    coefficient_m2,coefficient_m1,List_m2,List_m1,M2_size,M1_size,R2,R1,p_m2,q_m2,p_m1,q_m1,M2_N,M1_N,fimag_N,fimag_size,distance,edge_taper,Angle_taper,k=read_input(inputfile);
    # 2. build model;
    m2,m2_n,m2_dA,m1,m1_n,m1_dA,fimag,fimag_n,fimag_dA=model_ccat(coefficient_m2,List_m2,M2_size[0],M2_size[1],M2_N[0],M2_N[1],R2,
                                                                  coefficient_m1,List_m1,M1_size[0],M1_size[1],M1_N[0],M1_N[1],R1,
                                                                  fimag_size[0],fimag_size[1],fimag_N[0],fimag_N[1],
                                                                  ad_m2,ad_m1,p_m2,q_m2,p_m1,q_m1);
   
    # 3. prepared the matrixes for 4 receiver locations;
    print('0') 
    source=Coord();
    source0=np.genfromtxt(sourcefile+'/center.txt');
    source.x=source0[...,0];source.y=source0[...,1];source.z=source0[...,2];
    Matrix21_0,Matrix3_0,cosm2_i_0,cosm2_r_0,cosm1_i_0,cosm1_r_0,Field_s,Field_fimag,Field_m1,Field_m2_0,aperture,theta_m2_0,phi_m2_0=First_computing(m2,m2_n,m2_dA,m1,m1_n,m1_dA,fimag,fimag_n,fimag_dA,defocus0,source,k,Angle_taper,edge_taper,feed_rotation=dA_f0,zernike_order=0,theta_max=theta_max,Keepmatrix=True);
    Matrix21_0,Matrix3_0,cosm2_i_0,cosm2_r_0,cosm1_i_0,cosm1_r_0,Field_m2_0=DATA2CUDA(Matrix21_0,Matrix3_0,
                                                                                      cosm2_i_0,cosm2_r_0,
                                                                                      cosm1_i_0,cosm1_r_0,
                                                                                      Field_m2_0,DEVICE=DEVICE0);
   
    
    
    
    del(Field_s,Field_fimag,Field_m1);
    List_2,List_1,m2,m1,aperture,p_m2,q_m2,p_m1,q_m1=DATA2CUDA(List_m2,List_m1,m2,m1,aperture,p_m2,q_m2,p_m1,q_m1,DEVICE=DEVICE0);
    
    theta_max_x=theta_max/180*np.pi
    theta_max_y=(theta_max+0)/180*np.pi
    errorFn0=make_zernike(Zernike_order,theta_m2_0*np.cos(phi_m2_0)/theta_max_x,theta_m2_0*np.sin(phi_m2_0)/theta_max_y,dtype='torch');

    r=copy.copy(R1)
    def fitfuc(Para_A,Para_P,Para_f):        
        R0=fitting_func_zernik_Feed2(Matrix21_0,Matrix3_0,cosm2_i_0,cosm2_r_0,cosm1_i_0,cosm1_r_0,
                               Field_m2_0,Para_f,errorFn0,k,Para_A,Para_P,aperture/400);
       
 
        #R=T.cat((R0,R1,R2));        
        return R0; 
        
    return fitfuc;     


'''
7.3. function used to do the first precise calculation and also produce a new simple and more efficient forward function. this function is used to get the large-scale error caused by phase pattern of the feed horn.
'''    
def Make_fitfuc_zernike_Feed_3Beam(inputfile,sourcefile,ad_m2,ad_m1,defocus0,Zernike_order,dA_f0,dA_f1,dA_f2,theta_max=20):
    
    # 0. read the input parameters from the input files;
    coefficient_m2,coefficient_m1,List_m2,List_m1,M2_size,M1_size,R2,R1,p_m2,q_m2,p_m1,q_m1,M2_N,M1_N,fimag_N,fimag_size,distance,edge_taper,Angle_taper,k=read_input(inputfile);
    # 2. build model;
    m2,m2_n,m2_dA,m1,m1_n,m1_dA,fimag,fimag_n,fimag_dA=model_ccat(coefficient_m2,List_m2,M2_size[0],M2_size[1],M2_N[0],M2_N[1],R2,
                                                                  coefficient_m1,List_m1,M1_size[0],M1_size[1],M1_N[0],M1_N[1],R1,
                                                                  fimag_size[0],fimag_size[1],fimag_N[0],fimag_N[1],
                                                                  ad_m2,ad_m1,p_m2,q_m2,p_m1,q_m1);
   
    # 3. prepared the matrixes for 4 receiver locations;
    print('0') 
    source=Coord();
    source0=np.genfromtxt(sourcefile+'/center.txt');
    source.x=source0[...,0];source.y=source0[...,1];source.z=source0[...,2];
    Matrix21_0,Matrix3_0,cosm2_i_0,cosm2_r_0,cosm1_i_0,cosm1_r_0,Field_s,Field_fimag,Field_m1,Field_m2_0,aperture,theta_m2_0,phi_m2_0=First_computing(m2,m2_n,m2_dA,m1,m1_n,m1_dA,fimag,fimag_n,fimag_dA,defocus0,source,k,Angle_taper,edge_taper,feed_rotation=dA_f0,zernike_order=0,theta_max=theta_max,Keepmatrix=True);
    Matrix21_0,Matrix3_0,cosm2_i_0,cosm2_r_0,cosm1_i_0,cosm1_r_0,Field_m2_0=DATA2CUDA(Matrix21_0,Matrix3_0,
                                                                                      cosm2_i_0,cosm2_r_0,
                                                                                      cosm1_i_0,cosm1_r_0,
                                                                                      Field_m2_0,DEVICE=DEVICE0);
   
    
    
    
    del(Field_s,Field_fimag,Field_m1);
    List_2,List_1,m2,m1,aperture,p_m2,q_m2,p_m1,q_m1=DATA2CUDA(List_m2,List_m1,m2,m1,aperture,p_m2,q_m2,p_m1,q_m1,DEVICE=DEVICE0);
    
    theta_max_x=theta_max/180*np.pi
    theta_max_y=(theta_max+0)/180*np.pi
    errorFn0=make_zernike(Zernike_order,theta_m2_0*np.cos(phi_m2_0)/theta_max_x,theta_m2_0*np.sin(phi_m2_0-dA_f0)/theta_max_y,dtype='torch');
    errorFn1=make_zernike(Zernike_order,theta_m2_0*np.cos(phi_m2_0)/theta_max_x,theta_m2_0*np.sin(phi_m2_0-dA_f1)/theta_max_y,dtype='torch');
    errorFn2=make_zernike(Zernike_order,theta_m2_0*np.cos(phi_m2_0)/theta_max_x,theta_m2_0*np.sin(phi_m2_0-dA_f2)/theta_max_y,dtype='torch');

    
    def fitfuc(Para_A,Para_P,Para_f):        
        R0=fitting_func_zernik_Feed2(Matrix21_0,Matrix3_0,cosm2_i_0,cosm2_r_0,cosm1_i_0,cosm1_r_0,
                               Field_m2_0,Para_f,errorFn0,k,Para_A[0:6],Para_P[0:5],aperture/400);
        
        R1=fitting_func_zernik_Feed2(Matrix21_0,Matrix3_0,cosm2_i_0,cosm2_r_0,cosm1_i_0,cosm1_r_0,
                               Field_m2_0,Para_f,errorFn1,k,Para_A[6:6*2],Para_P[5*1:5*2],aperture/400);
        
        R2=fitting_func_zernik_Feed2(Matrix21_0,Matrix3_0,cosm2_i_0,cosm2_r_0,cosm1_i_0,cosm1_r_0,
                               Field_m2_0,Para_f,errorFn2,k,Para_A[6*2:],Para_P[5*2:],aperture/400);
       
 
        R=T.cat((R0,R1,R2));        
        return R; 
        
    return fitfuc;     






