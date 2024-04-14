import os

import numpy as np
import scipy.optimize
import torch as T
import time
import h5py

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pyvista as pv
pv.set_jupyter_backend('trame')#('static')#


from mirrorpy import profile,squarepanel,deformation,ImagPlane,adjuster,parallelogram_panel
# 'profile' with 2d poynomial coefficients produces surface function f(x,y).
# 'squarepanel' is the rim of the panels of mirrors and sampling range. 
# 'deformation' is the panel surface changes caused by errors of panel adjusters.
# 'ImagPlane' is used to define the sampling range of IF field plane.
from coordinate_operations import Coord
from coordinate_operations import Transform_local2global as local2global
# geometry models are smampled by discrete points. Coordinates of the points 
# are expressed by Coord.x, Coord.y, Coord.z
# other functions are known from theirs names used for transformations between coordinate
# systems by giving their origin point displacement and rotation angles.
from Kirchhoff import Complex#, PO_scalar
from KirchhoffpyGPU import PO_scalar
# PO_scalar is the field solver. Compex is used to express complex fields on mirror surfaces
# and desired field region.
from Feedpy import Gaussibeam
# Input field from a Gaussian feed horn.

from zernike_torch import mkCFn as make_zernike;
from zernike_torch import N as poly_N;

from inference import DATA2TORCH, correctphase2
from inference import fitting_func, fitting_func_zernike

from Pyccat import CCAT_holo,Load_Mat

# ploting package
from pyplot import plot_beamcontour


# surface errors expressed by zernike polynomials.
c=299792458*1000

holo_setup={'Rx1':([0,0,600],'scan/on-axis.txt'),
            'Rx2':([400,400,600],'scan/400_400_600.txt'),
            'Rx3':([400,-400,600],'scan/400_-400_600.txt'),
            'Rx4':([-400,400,600],'scan/-400_400_600.txt'),
            'Rx5':([-400,-400,600],'scan/-400_-400_600.txt')
            }
class CCAT_holo_half(CCAT_holo):
    def __init__(self,
                 Model_folder,
                 output_folder,
                 holo_conf=holo_setup,
                 input_Rx_beam=Gaussibeam):
        
        CCAT_holo.__init__(self,Model_folder,
                           output_folder,
                           holo_conf=holo_conf,
                           input_Rx_beam=input_Rx_beam)
        # get carbon fibon plate by giving the center position and four corners locations.
        self.M2_CF_4points=np.genfromtxt(Model_folder+'/CF_m2.txt',delimiter=',')
        self.M1_CF_4points=np.genfromtxt(Model_folder+'/CF_m1.txt',delimiter=',')
        self.Ox1=self.M1_CF_4points[0,0]
        self.Oy1=self.M1_CF_4points[0,1]

        self.Ox2=self.M2_CF_4points[0,0]
        self.Oy2=self.M2_CF_4points[0,1]

        self.v1a=self.M1_CF_4points[1,:]-self.M1_CF_4points[0,:]
        self.v1b=self.M1_CF_4points[3,:]-self.M1_CF_4points[0,:]

        self.v2a=self.M2_CF_4points[1,:]-self.M2_CF_4points[0,:]
        self.v2b=self.M2_CF_4points[3,:]-self.M2_CF_4points[0,:]

        parameters=np.genfromtxt(Model_folder+'/Model.txt',delimiter=',')[:,1]

        self.CF2_N=parameters[20:22].astype(int)
        self.CF1_N=parameters[22:24].astype(int)
        self.CF2_dz=parameters[18]
        self.CF1_dz=parameters[19]
        
        self.fimag2_N=parameters[24:26]
        self.fimag2_size=parameters[26:]
        '''surface'''
        # define surface profile of M1 and M2 in their local coordinates
        CF2_poly_coeff=np.genfromtxt(Model_folder+'/coeffi_m2CF.txt',delimiter=',')
        CF1_poly_coeff=np.genfromtxt(Model_folder+'/coeffi_m1CF.txt',delimiter=',')
        CF2_poly_coeff[0,0]=self.CF2_dz
        CF1_poly_coeff[0,0]=self.CF1_dz
        # the 2D polynomial surface  
        self.surface_m2_CF=profile(CF2_poly_coeff,1)
        self.surface_m1_CF=profile(CF1_poly_coeff,1)

        self.m2_CF,self.m2_CF_n,self.m2_CF_dA=parallelogram_panel(self.Ox2,self.Oy2,
                                                                  self.v2a,self.v2b,
                                                                  self.CF2_N[0], self.CF2_N[1],
                                                                  self.surface_m2_CF,
                                                                  quadrature='uniform')
        self.m1_CF,self.m1_CF_n,self.m1_CF_dA=parallelogram_panel(self.Ox1,self.Oy1,
                                                                  self.v1a,self.v1b,
                                                                  self.CF1_N[0], self.CF1_N[1],
                                                                  self.surface_m1_CF,
                                                                  quadrature='uniform')
        self.fimag2,self.fimag2_n,self.fimag2_dA=ImagPlane(self.fimag2_size[0],self.fimag2_size[1],
                                                        self.fimag2_N[0],self.fimag2_N[1]  
                                                        )

        self._coords2()
    def _coords2(self,Rx=[0,0,0]):
        '''coordinates systems'''
        '''
        #angle# is angle change of local coordinates and global coordinates;
        #D#     is the distance between origin of local coord and global coord in global coordinates;
        '''
        '''
        some germetrical parametrs
        '''

        #Theta_0  =  0.927295218001612; # offset angle of MR;
        #Ls       =  12000.0;           # distance between focal point and SR
        #Lm       =  6000.0;            # distance between MR and SR;
        #L_fimag  = 18000+Ls
        #F        = 20000               # equivalent focal length of M2
        data=np.genfromtxt(self.Model_folder+'/coord.txt',delimiter=',')[:,1]
        Theta_0=data[0]
        Ls=data[1]
        Lm=data[2]
        F=data[3]
        L_fimag=1/(1/Ls-1/F)
        
        

        self.angle_m2=[-(np.pi/2+Theta_0)/2,0,0] #  1. m2 and global co-ordinates
        self.D_m2=[0,-Lm*np.sin(Theta_0),0]
        
        self.angle_m1=[-Theta_0/2,0,0]          #  2. m1 and global co-ordinates
        self.D_m1=[0,0,Lm*np.cos(Theta_0)]
        
        self.angle_s=[0,np.pi,0];               #  3. source and global co-ordinates
        self.D_s=[0,0,0]
        
        self.angle_fimag=[-Theta_0,0,0];        #  4. fimag and global co-ordinates
        defocus_fimag=[0,0,0]
        defocus_fimag[2]=1/(1/F-1/(Ls+Rx[2]))+L_fimag
        defocus_fimag[1]=(F+L_fimag-defocus_fimag[2])/F*Rx[1]
        defocus_fimag[0]=(F+L_fimag-defocus_fimag[2])/F*Rx[0]
        self.D_fimag=[0,0,0]
        self.D_fimag[0]=defocus_fimag[0]
        self.D_fimag[1]=defocus_fimag[1]*np.cos(Theta_0)\
            -np.sin(Theta_0)*(L_fimag-defocus_fimag[2]+Lm)
        self.D_fimag[2]=-defocus_fimag[1]*np.sin(Theta_0)\
            -np.cos(Theta_0)*(L_fimag-defocus_fimag[2])
        
        defocus_fimag2=[0,0,0]
        defocus_fimag2[2]=Ls+Rx[2]
        defocus_fimag2[1]=Rx[1]
        defocus_fimag2[0]=Rx[0]

        self.D_fimag2=[0,0,0]
        self.D_fimag2[0]=Rx[0]
        L1=Lm*np.sin(Theta_0)-self.CF2_dz/np.cos(np.pi/4-Theta_0/2)
        L=L1+Rx[2]+(Ls-Lm*np.sin(Theta_0))
        self.D_fimag2[1]=-L1-L*np.sin(Theta_0)+defocus_fimag2[1]*np.cos(Theta_0)
        self.D_fimag2[2]=-L*np.cos(Theta_0)-defocus_fimag2[1]*np.sin(Theta_0)
        self.angle_fimag2=[-Theta_0,0,0]




        # feed coordinate system
        '''
        C=1/(1/Lm-1/F)+defocus[2]+Ls;
        C=21000;
        angle_f=[np.pi/2-defocus[1]/C,0,-defocus[0]/C]; 
        D_f=[defocus[0],Ls+defocus[2]-Lm*np.sin(Theta_0),-defocus[1]];
        '''
        self.angle_f=[np.pi/2,0,0];    
        self.D_f=[Rx[0],Ls+Rx[2]-Lm*np.sin(Theta_0),-Rx[1]] 

        
    def CF_view(self):
        CF2=Coord()
        CF2.x=self.M2_CF_4points[:,0]
        CF2.y=self.M2_CF_4points[:,1]
        CF2.z,Vn=self.surface_m2_CF(CF2.x,CF2.y)

        CF1=Coord()
        CF1.x=self.M1_CF_4points[:,0]
        CF1.y=self.M1_CF_4points[:,1]
        CF1.z,Vn=self.surface_m1_CF(CF1.x,CF1.y)  

        CF2=local2global(self.angle_m2,self.D_m2,CF2)
        CF1=local2global(self.angle_m1,self.D_m1,CF1)  
        print(CF2.x,CF2.y,CF2.z)
        print(CF1.x,CF1.y,CF1.z)

        points2 = np.c_[CF2.x.reshape(-1), CF2.y.reshape(-1), CF2.z.reshape(-1)]
        points1 = np.c_[CF1.x.reshape(-1), CF1.y.reshape(-1), CF1.z.reshape(-1)]
        #print(points2)

        faces2=np.ones(1).astype(int)*4
        factor=np.linspace(0,3,4).astype(int).reshape(-1,4)
        faces2=np.c_[faces2,factor].ravel()

        faces1=np.ones(1).astype(int)*4
        factor=np.linspace(0,3,4).astype(int).reshape(-1,4)
        faces1=np.c_[faces1,factor]

        
        CF1 = pv.PolyData(points1,faces1)
        CF2 = pv.PolyData(points2,faces2)

        
        IF2=Coord()
        IF2.x=np.array([500,500,-500,-500])
        IF2.y=np.array([500,-500,-500,500])
        IF2.z=np.array([0,0,0,0])
        IF2=local2global(self.angle_fimag,self.D_fimag2,IF2)
        points3=np.c_[IF2.x.reshape(-1),
                      IF2.y.reshape(-1),
                      IF2.z.reshape(-1)]
        
        faces3=np.ones(1).astype(int)*4
        factor=np.linspace(0,3,4).astype(int).reshape(-1,4)
        faces3=np.c_[faces3,factor].ravel()
        IF2 = pv.PolyData(points3,faces3)

        self.widget.add_mesh(CF1,show_edges=True,color="gray")
        self.widget.add_mesh(CF2,show_edges=True,color="gray")
        #self.widget.add_mesh(IF2,show_edges=True)#,color="gray")

    def _beamA(self,scan_file,Rx=[0,0,0],
              Matrix=False,
              S2_init=np.zeros((5,69)),S1_init=np.zeros((5,77)),Error_m2=0,Error_m1=0,file_name='data'):
        '''**scan_file** is scan trajectary data;
           **   Rx    ** position of receiver in focal plane or receiver plane
           ** S2_init ** Offset of panel adjusters on M2
           ** S1_init ** Offset of panel adjusters on M1
           ** Error_2/1** Self-defined surface errors on M2 and M1.
        '''
        Feed_beam=self.input_feed_beam
        trace=np.genfromtxt(scan_file,delimiter=',')
        scan_pattern=Coord()
        scan_pattern.x=trace[:,0]
        scan_pattern.y=trace[:,1]
        scan_pattern.z=trace[:,2]

        ''' first beam pattern calculation'''
        
        # Set receiver location
        self._coords2(Rx=Rx)

        # create M1 M2 & IF plane model.
        self.m2,self.m2_n,self.m2_dA=squarepanel(self.Panel_center_M2[...,0],self.Panel_center_M2[...,1],
                                                    self.M2_size[0],self.M2_size[1],
                                                    self.M2_N[0],self.M2_N[1],
                                                    self.surface_m2
                                                    )
        self.m1,self.m1_n,self.m1_dA=squarepanel(self.Panel_center_M1[...,0],self.Panel_center_M1[...,1],
                                                    self.M1_size[0],self.M1_size[1],
                                                    self.M1_N[0],self.M1_N[1],
                                                    self.surface_m1
                                                    )
        
        self.fimag,self.fimag_n,self.fimag_dA=ImagPlane(self.fimag_size[0],self.fimag_size[1],
                                                        self.fimag_N[0],self.fimag_N[1]  
                                                        )
        self.m2_CF,self.m2_CF_n,self.m2_CF_dA=parallelogram_panel(self.Ox2,self.Oy2,
                                                                  self.v2a,self.v2b,
                                                                  self.CF2_N[0], self.CF2_N[1],
                                                                  self.surface_m2_CF,
                                                                  quadrature='uniform')
        self.m1_CF,self.m1_CF_n,self.m1_CF_dA=parallelogram_panel(self.Ox1,self.Oy1,
                                                                  self.v1a,self.v1b,
                                                                  self.CF1_N[0], self.CF1_N[1],
                                                                  self.surface_m1_CF,
                                                                  quadrature='uniform')
        self.fimag2,self.fimag2_n,self.fimag2_dA=ImagPlane(self.fimag2_size[0],self.fimag2_size[1],
                                                        self.fimag2_N[0],self.fimag2_N[1]  
                                                        )


        # Misalignment of panel adjusters
        self.m2.z=self.m2.z + Error_m2 + deformation(S2_init.ravel(),
                                          self.Panel_center_M2,
                                          self.p_m2,self.q_m2,self.m2)
        self.m1.z=self.m1.z + Error_m1 - deformation(S1_init.ravel(),
                                          self.Panel_center_M1,
                                          self.p_m1,self.q_m1,self.m1)

        start=time.perf_counter()
        # 1. convert MIRROR 2 into global coordinate system
        m2=local2global(self.angle_m2,self.D_m2,self.m2)
        m2_n=local2global(self.angle_m2,[0,0,0],self.m2_n)
        Field_m2=Complex()

        # 2. illumination field on M2
        print('step 1:\n')
        Field_m2.real,Field_m2.imag,cosm2_i=Feed_beam(self.edge_taper,
                                                      self.Angle_taper,
                                                      self.k,
                                                      m2,m2_n,
                                                      self.angle_f,self.D_f
                                                      )
        
        # 3. calculate field on IF plane
        print('step 2:\n')
        fimag=local2global(self.angle_fimag,self.D_fimag,self.fimag)
        fimag_n=local2global(self.angle_fimag,[0,0,0],self.fimag_n)
        Matrix1,self.Field_fimag,cosm2_r=PO_scalar(m2,m2_n,
                                              self.m2_dA,
                                              fimag,cosm2_i,
                                              Field_m2,-self.k,
                                              Keepmatrix=Matrix
                                              )
        
        # 4. calculate field on M1
        m1=local2global(self.angle_m1,self.D_m1,self.m1)
        m1_n=local2global(self.angle_m1,[0,0,0],self.m1_n)
        self.aperture_xy=np.append(m1.x,m1.y).reshape(2,-1)/self.R1

        print(self.m1_CF.x.max())
        m1_CF=local2global(self.angle_m1,self.D_m1,self.m1_CF)
        m1_CF_n=local2global(self.angle_m1,[0,0,0],self.m1_CF_n)

        NN=int(fimag.x.size/2)
        Fimag0=[fimag.x[NN],fimag.y[NN],fimag.z[NN]]
        x=Fimag0[0].item()-m1.x.reshape(1,-1)
        y=Fimag0[1].item()-m1.y.reshape(1,-1)
        z=Fimag0[2].item()-m1.z.reshape(1,-1)
        
        r=np.sqrt(x**2+y**2+z**2)
        cosm1_i=(x*m1_n.x+y*m1_n.y+z*m1_n.z)/r

        NN=int(fimag.x.size/2)
        Fimag0=[fimag.x[NN],fimag.y[NN],fimag.z[NN]]
        x=Fimag0[0].item()-m1_CF.x.reshape(1,-1)
        y=Fimag0[1].item()-m1_CF.y.reshape(1,-1)
        z=Fimag0[2].item()-m1_CF.z.reshape(1,-1)
        
        r=np.sqrt(x**2+y**2+z**2)
        cosm1_CF_i=(x*m1_CF_n.x+y*m1_CF_n.y+z*m1_CF_n.z)/r
        #cosm1_i=T.tensor(cosm1_i).to(DEVICE);
        del(x,y,z,r)
        print('step 3:\n')
        Matrix2,Field_m1,cosm=PO_scalar(fimag,fimag_n,self.fimag_dA,
                                        m1,np.array([1]),
                                        self.Field_fimag,
                                        self.k,
                                        Keepmatrix=Matrix
                                        )
        del(cosm)
        Matrix2,Field_m1_CF,cosm=PO_scalar(fimag,fimag_n,self.fimag_dA,
                                        m1_CF,np.array([1]),
                                        self.Field_fimag,
                                        self.k,
                                        Keepmatrix=Matrix
                                        )
        del(cosm)
        #5. calculate the field in the source range;
        source=local2global(self.angle_s,self.D_s,scan_pattern)
        print('step 4:\n')
        Matrix3,self.Field_s1,cosm1_r=PO_scalar(m1,m1_n,self.m1_dA,
                                          source,cosm1_i,
                                          Field_m1,
                                          self.k,
                                          Keepmatrix=Matrix
                                          )
        
        Matrix3,self.Field_s2,cosm1_r=PO_scalar(m1_CF,m1_CF_n,self.m1_CF_dA,
                                          source,cosm1_CF_i,
                                          Field_m1_CF,
                                          self.k,
                                          Keepmatrix=Matrix
                                          )
        self.Field_s=Complex()
        self.Field_s.real=self.Field_s1.real+self.Field_s2.real
        self.Field_s.imag=self.Field_s1.imag+self.Field_s2.imag
        
        elapsed =(time.perf_counter()-start)
        print('time used:',elapsed)
        # Save the computation data into h5py file, and the intermediate Matrixs that wil
        # be used for accelerating the forward beam calculations.
        self.output_filename=self.output_folder+'/'+file_name+'_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
        with h5py.File(self.output_filename,'w') as f:
              #f.create_dataset('conf',data=(Rx,scan_file))
              f.create_dataset('freq (GHz)',data=self.freq)
              f.create_dataset('F_m2_real',data=Field_m2.real)
              f.create_dataset('F_m2_imag',data=Field_m2.imag)

              f.create_dataset('F_if_real',data=self.Field_fimag.real)
              f.create_dataset('F_if_imag',data=self.Field_fimag.imag)

              f.create_dataset('FA_m1_real',data=Field_m1.real)
              f.create_dataset('FA_m1_imag',data=Field_m1.imag)

              f.create_dataset('FA_m1_CF_real',data=Field_m1_CF.real)
              f.create_dataset('FA_m1_CF_imag',data=Field_m1_CF.imag)

              f.create_dataset('FA_beam_real',data=self.Field_s.real)
              f.create_dataset('FA_beam_imag',data=self.Field_s.imag)

              f.create_dataset('scan_pattern',data=np.concatenate((source.x,
                                                                   source.y,
                                                                   source.z)).reshape(3,-1))
    def _beamB(self,scan_file,Rx=[0,0,0],
              Matrix=False,
              S2_init=np.zeros((5,69)),S1_init=np.zeros((5,77)),Error_m2=0,Error_m1=0,file_name='data'):
        '''**scan_file** is scan trajectary data;
           **   Rx    ** position of receiver in focal plane or receiver plane
           ** S2_init ** Offset of panel adjusters on M2
           ** S1_init ** Offset of panel adjusters on M1
           ** Error_2/1** Self-defined surface errors on M2 and M1.
        '''
        Feed_beam=self.input_feed_beam
        trace=np.genfromtxt(scan_file,delimiter=',')
        scan_pattern=Coord()
        scan_pattern.x=trace[:,0]
        scan_pattern.y=trace[:,1]
        scan_pattern.z=trace[:,2]

        ''' first beam pattern calculation'''
        
        # Set receiver location
        self._coords2(Rx=Rx)

        # create M1 M2 & IF plane model.
        self.m2,self.m2_n,self.m2_dA=squarepanel(self.Panel_center_M2[...,0],self.Panel_center_M2[...,1],
                                                    self.M2_size[0],self.M2_size[1],
                                                    self.M2_N[0],self.M2_N[1],
                                                    self.surface_m2
                                                    )
        self.m1,self.m1_n,self.m1_dA=squarepanel(self.Panel_center_M1[...,0],self.Panel_center_M1[...,1],
                                                    self.M1_size[0],self.M1_size[1],
                                                    self.M1_N[0],self.M1_N[1],
                                                    self.surface_m1
                                                    )
        
        self.fimag,self.fimag_n,self.fimag_dA=ImagPlane(self.fimag_size[0],self.fimag_size[1],
                                                        self.fimag_N[0],self.fimag_N[1]  
                                                        )
        self.m2_CF,self.m2_CF_n,self.m2_CF_dA=parallelogram_panel(self.Ox2,self.Oy2,
                                                                  self.v2a,self.v2b,
                                                                  self.CF2_N[0], self.CF2_N[1],
                                                                  self.surface_m2_CF,
                                                                  quadrature='uniform')
        self.m1_CF,self.m1_CF_n,self.m1_CF_dA=parallelogram_panel(self.Ox1,self.Oy1,
                                                                  self.v1a,self.v1b,
                                                                  self.CF1_N[0], self.CF1_N[1],
                                                                  self.surface_m1_CF,
                                                                  quadrature='uniform')
        self.fimag2,self.fimag2_n,self.fimag2_dA=ImagPlane(self.fimag2_size[0],self.fimag2_size[1],
                                                        self.fimag2_N[0],self.fimag2_N[1]  
                                                        )


        # Misalignment of panel adjusters
        self.m2.z=self.m2.z + Error_m2 + deformation(S2_init.ravel(),
                                          self.Panel_center_M2,
                                          self.p_m2,self.q_m2,self.m2)
        self.m1.z=self.m1.z + Error_m1 - deformation(S1_init.ravel(),
                                          self.Panel_center_M1,
                                          self.p_m1,self.q_m1,self.m1)

        start=time.perf_counter()
        # 1. convert MIRROR 2 into global coordinate system
        m2_CF=local2global(self.angle_m2,self.D_m2,self.m2_CF)
        m2_CF_n=local2global(self.angle_m2,[0,0,0],self.m2_CF_n)
        Field_m2_CF=Complex()

        # 2. illumination field on M2
        print('step 1:\n')
        Field_m2_CF.real,Field_m2_CF.imag,cosm2_CF_i=Feed_beam(self.edge_taper,
                                                      self.Angle_taper,
                                                      self.k,
                                                      m2_CF,m2_CF_n,
                                                      self.angle_f,self.D_f
                                                      )
        
        # 3. calculate field on IF plane
        print('step 2:\n')
        fimag2=local2global(self.angle_fimag2,self.D_fimag2,self.fimag2)
        fimag2_n=local2global(self.angle_fimag2,[0,0,0],self.fimag2_n)
        Matrix1,self.Field_fimag2,cosm2_CF_r=PO_scalar(m2_CF,m2_CF_n,
                                              self.m2_CF_dA,
                                              fimag2,cosm2_CF_i,
                                              Field_m2_CF,-self.k,
                                              Keepmatrix=Matrix
                                              )
        
        # 4. calculate field on M1
        m1=local2global(self.angle_m1,self.D_m1,self.m1)
        m1_n=local2global(self.angle_m1,[0,0,0],self.m1_n)

        m1_CF=local2global(self.angle_m1,self.D_m1,self.m1_CF)
        m1_CF_n=local2global(self.angle_m1,[0,0,0],self.m1_CF_n)

        NN=int(fimag2.x.size/2)
        Fimag0=[fimag2.x[NN],fimag2.y[NN],fimag2.z[NN]]
        x=Fimag0[0].item()-m1.x.reshape(1,-1)
        y=Fimag0[1].item()-m1.y.reshape(1,-1)
        z=Fimag0[2].item()-m1.z.reshape(1,-1)
        
        r=np.sqrt(x**2+y**2+z**2)
        cosm1_i=(x*m1_n.x+y*m1_n.y+z*m1_n.z)/r
        print(cosm1_i)
        #cosm1_i=T.tensor(cosm1_i).to(DEVICE);


        NN=int(fimag2.x.size/2)
        Fimag0=[fimag2.x[NN],fimag2.y[NN],fimag2.z[NN]]
        x=Fimag0[0].item()-m1_CF.x.reshape(1,-1)
        y=Fimag0[1].item()-m1_CF.y.reshape(1,-1)
        z=Fimag0[2].item()-m1_CF.z.reshape(1,-1)
        
        r=np.sqrt(x**2+y**2+z**2)
        cosm1_CF_i=(x*m1_CF_n.x+y*m1_CF_n.y+z*m1_CF_n.z)/r
        print(cosm1_CF_i)

        del(x,y,z,r)
        print('step 3:\n')
        Matrix2,Field_m1,cosm=PO_scalar(fimag2,fimag2_n,self.fimag2_dA,
                                        m1,np.array([1]),
                                        self.Field_fimag2,
                                        self.k,
                                        Keepmatrix=Matrix
                                        )
        del(cosm)
        Matrix2,Field_m1_CF,cosm=PO_scalar(fimag2,fimag2_n,self.fimag2_dA,
                                        m1_CF,np.array([1]),
                                        self.Field_fimag2,
                                        self.k,
                                        Keepmatrix=Matrix
                                        )
        del(cosm)

        '''
        emerging m1 and m2 to m12
        '''    
        Matrix21=Complex()
        if Matrix:
            Matrix21.real=np.matmul(Matrix2.real,Matrix1.real)-np.matmul(Matrix2.imag,Matrix1.imag)
            Matrix21.imag=np.matmul(Matrix2.real,Matrix1.imag)+np.matmul(Matrix2.imag,Matrix1.real)
        else:
            pass
        del(Matrix2,Matrix1)
        
        #5. calculate the field in the source range;
        source=local2global(self.angle_s,self.D_s,scan_pattern)
        print('step 4:\n')
        Matrix3,self.Field_s1,cosm1_r=PO_scalar(m1,m1_n,self.m1_dA,
                                          source,cosm1_i,
                                          Field_m1,
                                          self.k,
                                          Keepmatrix=Matrix
                                          )
        
        Matrix3,self.Field_s2,cosm1_r=PO_scalar(m1_CF,m1_CF_n,self.m1_CF_dA,
                                          source,cosm1_CF_i,
                                          Field_m1_CF,
                                          self.k,
                                          Keepmatrix=Matrix
                                          )
        self.Field_s=Complex()
        self.Field_s.real=self.Field_s1.real+self.Field_s2.real
        self.Field_s.imag=self.Field_s1.imag+self.Field_s2.imag
        
        elapsed =(time.perf_counter()-start)
        print('time used:',elapsed)
        # Save the computation data into h5py file, and the intermediate Matrixs that wil
        # be used for accelerating the forward beam calculations.
        self.output_filename=self.output_folder+'/'+file_name+'_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
        with h5py.File(self.output_filename,'a') as f:
              #f.create_dataset('conf',data=(Rx,scan_file))
              #f.create_dataset('freq (GHz)',data=self.freq)
              f.create_dataset('F_m2_CF_real',data=Field_m2_CF.real)
              f.create_dataset('F_m2_CF_imag',data=Field_m2_CF.imag)

              f.create_dataset('F_if2_real',data=self.Field_fimag2.real)
              f.create_dataset('F_if2_imag',data=self.Field_fimag2.imag)

              f.create_dataset('FB_m1_real',data=Field_m1.real)
              f.create_dataset('FB_m1_imag',data=Field_m1.imag)

              f.create_dataset('FB_m1_CF_real',data=Field_m1_CF.real)
              f.create_dataset('FB_m1_CF_imag',data=Field_m1_CF.imag)

              f.create_dataset('FB_beam_real',data=self.Field_s.real)
              f.create_dataset('FB_beam_imag',data=self.Field_s.imag)
              
        with h5py.File(self.output_filename,'r') as f:
            F_s_real=self.Field_s.real+f['FA_beam_real'][:]
            F_s_imag=self.Field_s.imag+f['FA_beam_imag'][:]
            F_m1_real=Field_m1.real+f['FA_m1_real'][:]
            F_m1_imag=Field_m1.imag+f['FA_m1_imag'][:]
            F_m1_CF_real=Field_m1_CF.real+f['FA_m1_CF_real'][:]
            F_m1_CF_imag=Field_m1_CF.imag+f['FA_m1_CF_imag'][:]
        with h5py.File(self.output_filename,'a') as f:
            f.create_dataset('F_m1_real',data=F_m1_real)
            f.create_dataset('F_m1_imag',data=F_m1_imag)
            f.create_dataset('F_m1_CF_real',data=F_m1_CF_real)
            f.create_dataset('F_m1_CF_imag',data=F_m1_CF_imag)
            f.create_dataset('F_beam_real',data=F_s_real)
            f.create_dataset('F_beam_imag',data=F_s_imag)

    def First_Beam_cal(self,S2_init=np.zeros((5,69)),
                       S1_init=np.zeros((5,77)),
                       Error_m2=0,Error_m1=0,Matrix=True):
        '''Set the holographic design and make the first beam calculations'''
        if self.holo_conf==None:
            print('set up the holographic configuration, e.g. Rx positions & the related scanning tracjectory!')
            pass
        else:
            print('The holographic setup:')
            for keys in self.holo_conf:
                print(keys,':',self.holo_conf[keys][0],self.holo_conf[keys][1])

            print('\n***Start the initial beam calculations ')
            print('***and prepare the required Matrixes used to speed up the forward beam calculations.')
            for keys in self.holo_conf:
                print(keys,':',self.holo_conf[keys][0],self.holo_conf[keys][1])
                self._beamA(self.holo_conf[keys][1],Rx=self.holo_conf[keys][0],Matrix=Matrix,S2_init=S2_init,S1_init=S1_init,Error_m2=Error_m2,Error_m1=Error_m1)
                self._beamB(self.holo_conf[keys][1],Rx=self.holo_conf[keys][0],Matrix=Matrix,S2_init=S2_init,S1_init=S1_init,Error_m2=Error_m2,Error_m1=Error_m1)
    
    def plot_beamA(self,filename=None):
        '''plot the lastest calculated beam'''
        if filename==None:
            filename=self.output_filename
        else:
            pass
        if filename==None:
            print('No input data!!!')
            pass
        else:
            print('Beam Rx: '+filename)
            with h5py.File(filename,'r') as f:
                beam=f['FA_beam_real'][:]+1j*f['FA_beam_imag'][:]
                F_M1=f['FA_m1_real'][:]+1j*f['FA_m1_imag'][:]
                F_M1_CF=f['FA_m1_CF_real'][:]+1j*f['FA_m1_CF_imag'][:]
                F_IF=f['F_if_real'][:]+1j*f['F_if_imag'][:]
                F_M2=f['F_m2_real'][:]+1j*f['F_m2_imag'][:]
                F_M2_CF=f['F_m2_CF_real'][:]+1j*f['F_m2_CF_imag'][:]
                
                NN=int(np.sqrt(f['scan_pattern'][0,:].size))
                X=f['scan_pattern'][0,:].reshape(NN,-1)
                Y=f['scan_pattern'][1,:].reshape(NN,-1)

            # Field on M2
            fig, axs = plt.subplots(1, 1, figsize=(12, 5))
            cmap='jet'
            M2_panelN=int(self.Panel_center_M2.size/2)
            Nx=self.M2_N[0]
            Ny=self.M2_N[1]
            N=Nx*Ny
            vmax=20*np.log10(np.abs(F_M2).max())
            vmin=vmax-15
            for n in range(M2_panelN):
                x=self.m2_0.x[n*N:(n+1)*N].reshape(Ny,Nx)
                y=self.m2_0.y[n*N:(n+1)*N].reshape(Ny,Nx)
                p1=axs.pcolor(x,y,
                                 20*np.log10(np.abs(F_M2[n*N:(n+1)*N].reshape(Ny,Nx))),
                                 cmap=cmap,vmin=vmin,vmax=vmax)
            Nx=self.CF2_N[0]
            Ny=self.CF2_N[1]
            x=self.m2_CF.x.reshape(Nx,Ny)
            y=self.m2_CF.y.reshape(Nx,Ny)
            p1=axs.pcolor(x,y,
                          20*np.log10(np.abs(F_M2_CF.reshape(Ny,Nx))),
                          cmap=cmap,vmin=vmin,vmax=vmax)
            axs.axis('equal')
            plt.show()
            print('M2 power:',(np.abs(F_M2)**2).sum()*self.m2_dA)

            # Field on IF
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            cmap='jet'
            x=self.fimag.x.reshape(int(self.fimag_N[1]),int(self.fimag_N[0]))
            y=self.fimag.y.reshape(int(self.fimag_N[1]),int(self.fimag_N[0]))
            F_IF=F_IF.reshape(int(self.fimag_N[1]),int(self.fimag_N[0]))
            p1=axs[0].pcolor(x[0,:],y[:,0],20*np.log10(np.abs(F_IF)),cmap='jet')
            axs[0].axis('equal')
            p2=axs[1].pcolor(x,y,np.angle(F_IF)*180/np.pi,cmap='jet',vmax=180,vmin=-180)
            axs[1].axis('equal')
            plt.show()
            print('IF1 power:',(np.abs(F_IF)**2).sum()*self.fimag_dA)

            # Fields on M1
            fig, axs = plt.subplots(1, 1, figsize=(12, 5))
            cmap='jet'
            M1_panelN=int(self.Panel_center_M1.size/2)
            Nx=self.M1_N[0]
            Ny=self.M1_N[1]
            N=Nx*Ny
            vmax=20*np.log10(np.abs(F_M1).max())
            vmin=vmax-20
            for n in range(M1_panelN):
                x=self.m1_0.x[n*N:(n+1)*N].reshape(Ny,Nx)
                y=self.m1_0.y[n*N:(n+1)*N].reshape(Ny,Nx)
                p1=axs.pcolor(x,y,
                                 20*np.log10(np.abs(F_M1[n*N:(n+1)*N].reshape(Ny,Nx))),
                                 cmap=cmap,vmin=vmin,vmax=vmax)
            Nx=self.CF1_N[0]
            Ny=self.CF1_N[1]
            x=self.m1_CF.x.reshape(Nx,Ny)
            y=self.m1_CF.y.reshape(Nx,Ny)
            p1=axs.pcolor(x,y,
                          20*np.log10(np.abs(F_M1_CF.reshape(Ny,Nx))),
                          cmap=cmap,vmin=vmin,vmax=vmax)
            axs.axis('equal')
            #axs[1].axis('equal')
            plt.show()
            print('M1 power:',(np.abs(F_M1)**2).sum()*self.m1_dA)

            # beams
            beam=beam.reshape(NN,-1)
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            cmap='jet'
            p1=axs[0].pcolor(X,Y,20*np.log10(np.abs(beam)),cmap='jet')
            axs[0].axis('equal')
            p2=axs[1].pcolor(X,Y,np.angle(beam)*180/np.pi,cmap='jet',vmax=180,vmin=-180)
            axs[1].axis('equal')
            plt.show()
            print('beam power:',(np.abs(beam)**2).sum())

            # cut plot 
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            cmap='jet'
            NN0=np.where(np.abs(beam)==np.abs(beam).max())
            print(NN0[0],NN0[1],20*np.log10(np.abs(beam).max()))
            p1=axs[0].plot(X[NN0[0][0],:],20*np.log10(np.abs(beam[NN0[0][0],:])))
            p2=axs[1].plot(Y[:,NN0[1][0]],20*np.log10(np.abs(beam[:,NN0[1][0]])))
            plt.grid(axis='both')
            plt.show()

    def plot_beamB(self,filename=None):
        '''plot the lastest calculated beam'''
        if filename==None:
            filename=self.output_filename
        else:
            pass
        if filename==None:
            print('No input data!!!')
            pass
        else:
            print('Beam Rx: '+filename)
            with h5py.File(filename,'r') as f:
                beam=f['FB_beam_real'][:]+1j*f['FB_beam_imag'][:]
                F_M1=f['FB_m1_real'][:]+1j*f['FB_m1_imag'][:]
                F_M1_CF=f['FB_m1_CF_real'][:]+1j*f['FB_m1_CF_imag'][:]
                F_IF=f['F_if2_real'][:]+1j*f['F_if2_imag'][:]
                F_M2=f['F_m2_real'][:]+1j*f['F_m2_imag'][:]
                F_M2_CF=f['F_m2_CF_real'][:]+1j*f['F_m2_CF_imag'][:]
                NN=int(np.sqrt(f['scan_pattern'][0,:].size))
                X=f['scan_pattern'][0,:].reshape(NN,-1)
                Y=f['scan_pattern'][1,:].reshape(NN,-1)
            
            # Field on M2
            fig, axs = plt.subplots(1, 1, figsize=(12, 5))
            cmap='jet'
            M2_panelN=int(self.Panel_center_M2.size/2)
            Nx=self.M2_N[0]
            Ny=self.M2_N[1]
            N=Nx*Ny
            vmax=20*np.log10(np.abs(F_M2).max())
            vmin=vmax-15
            for n in range(M2_panelN):
                x=self.m2_0.x[n*N:(n+1)*N].reshape(Ny,Nx)
                y=self.m2_0.y[n*N:(n+1)*N].reshape(Ny,Nx)
                p1=axs.pcolor(x,y,
                                 20*np.log10(np.abs(F_M2[n*N:(n+1)*N].reshape(Ny,Nx))),
                                 cmap=cmap,vmin=vmin,vmax=vmax)
            Nx=self.CF2_N[0]
            Ny=self.CF2_N[1]
            x=self.m2_CF.x.reshape(Nx,Ny)
            y=self.m2_CF.y.reshape(Nx,Ny)
            p1=axs.pcolor(x,y,
                          20*np.log10(np.abs(F_M2_CF.reshape(Ny,Nx))),
                          cmap=cmap,vmin=vmin,vmax=vmax)
            axs.axis('equal')
            plt.show()
            print('M2_CF power:',(np.abs(F_M2_CF)**2).sum()*self.m2_CF_dA)

            # Field on IF
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            cmap='jet'
            x=self.fimag2.x.reshape(int(self.fimag2_N[1]),int(self.fimag2_N[0]))
            y=self.fimag2.y.reshape(int(self.fimag2_N[1]),int(self.fimag2_N[0]))
            F_IF=F_IF.reshape(int(self.fimag2_N[1]),int(self.fimag2_N[0]))
            p1=axs[0].pcolor(x[0,:],y[:,0],20*np.log10(np.abs(F_IF)),cmap='jet')
            axs[0].axis('equal')
            p2=axs[1].pcolor(x,y,np.angle(F_IF)*180/np.pi,cmap='jet',vmax=180,vmin=-180)
            axs[1].axis('equal')
            plt.show()
            print('IF2 power:',(np.abs(F_IF)**2).sum()*self.fimag2_dA)

            # Fields on M1
            fig, axs = plt.subplots(1, 1, figsize=(12, 5))
            cmap='jet'
            M1_panelN=int(self.Panel_center_M1.size/2)
            Nx=self.M1_N[0]
            Ny=self.M1_N[1]
            N=Nx*Ny
            vmax=20*np.log10(np.abs(F_M1_CF).max())
            vmin=vmax-20
            for n in range(M1_panelN):
                x=self.m1_0.x[n*N:(n+1)*N].reshape(Ny,Nx)
                y=self.m1_0.y[n*N:(n+1)*N].reshape(Ny,Nx)
                p1=axs.pcolor(x,y,
                                 20*np.log10(np.abs(F_M1[n*N:(n+1)*N].reshape(Ny,Nx))),
                                 cmap=cmap,vmin=vmin,vmax=vmax)
            Nx=self.CF1_N[0]
            Ny=self.CF1_N[1]
            x=self.m1_CF.x.reshape(Nx,Ny)
            y=self.m1_CF.y.reshape(Nx,Ny)
            p1=axs.pcolor(x,y,
                          20*np.log10(np.abs(F_M1_CF.reshape(Ny,Nx))),
                          cmap=cmap,vmin=vmin,vmax=vmax)
            print(np.abs(F_M1_CF).max())

            axs.axis('equal')
            #axs[1].axis('equal')
            plt.show()
            print('M1 CF power:',(np.abs(F_M1_CF)**2).sum()*self.m1_CF_dA)

            # beam
            beam=beam.reshape(NN,-1)
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            cmap='jet'
            p1=axs[0].pcolor(X,Y,20*np.log10(np.abs(beam)),cmap='jet')
            axs[0].axis('equal')
            p2=axs[1].pcolor(X,Y,np.angle(beam)*180/np.pi,cmap='jet',vmax=180,vmin=-180)
            axs[1].axis('equal')
            plt.show()

            # cut plot 
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            cmap='jet'
            NN0=np.where(np.abs(beam)==np.abs(beam).max())
            print(NN0[0],NN0[1],20*np.log10(np.abs(beam).max()))
            p1=axs[0].plot(X[NN0[0][0],:],20*np.log10(np.abs(beam[NN0[0][0],:])))
            p2=axs[1].plot(Y[:,NN0[1][0]],20*np.log10(np.abs(beam[:,NN0[1][0]])))
            plt.grid(axis='both')
            plt.show()

    def plot_beam(self,filename=None):
        '''plot the lastest calculated beam'''
        if filename==None:
            filename=self.output_filename
        else:
            pass
        if filename==None:
            print('No input data!!!')
            pass
        else:
            print('Beam Rx: '+filename)
            with h5py.File(filename,'r') as f:
                beam=f['F_beam_real'][:]+1j*f['F_beam_imag'][:]
                F_M1=f['F_m1_real'][:]+1j*f['F_m1_imag'][:]
                F_M1_CF=f['F_m1_CF_real'][:]+1j*f['F_m1_CF_imag'][:]
                F_IF=f['F_if2_real'][:]+1j*f['F_if2_imag'][:]
                F_M2=f['F_m2_real'][:]+1j*f['F_m2_imag'][:]
                F_M2_CF=f['F_m2_CF_real'][:]+1j*f['F_m2_CF_imag'][:]
                NN=int(np.sqrt(f['scan_pattern'][0,:].size))
                x=f['scan_pattern'][0,:].reshape(NN,-1)
                y=f['scan_pattern'][1,:].reshape(NN,-1)
            beam=beam.reshape(NN,-1)
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            cmap='jet'
            p1=axs[0].pcolor(x,y,20*np.log10(np.abs(beam)),cmap='jet')
            axs[0].axis('equal')
            p2=axs[1].pcolor(x,y,np.angle(beam)*180/np.pi,cmap='jet',vmax=180,vmin=-180)
            axs[1].axis('equal')
            plt.show()

            # cut plot 
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            cmap='jet'
            NN0=np.where(np.abs(beam)==np.abs(beam).max())
            print(NN0[0],NN0[1],20*np.log10(np.abs(beam).max()))
            p1=axs[0].plot(x[NN0[0][0],:],20*np.log10(np.abs(beam[NN0[0][0],:])))
            p2=axs[1].plot(y[:,NN0[1][0]],20*np.log10(np.abs(beam[:,NN0[1][0]])))
            plt.grid(axis='both')
            plt.show()

            # Fields on M1
            fig, axs = plt.subplots(1, 1, figsize=(12, 5))
            cmap='jet'
            M1_panelN=int(self.Panel_center_M1.size/2)
            Nx=self.M1_N[0]
            Ny=self.M1_N[1]
            N=Nx*Ny
            vmax=20*np.log10(np.abs(F_M1_CF).max())
            vmin=vmax-20
            for n in range(M1_panelN):
                x=self.m1_0.x[n*N:(n+1)*N].reshape(Ny,Nx)
                y=self.m1_0.y[n*N:(n+1)*N].reshape(Ny,Nx)
                p1=axs.pcolor(x,y,
                                 20*np.log10(np.abs(F_M1[n*N:(n+1)*N].reshape(Ny,Nx))),
                                 cmap=cmap,vmin=vmin,vmax=vmax)
            Nx=self.CF1_N[0]
            Ny=self.CF1_N[1]
            x=self.m1_CF.x.reshape(Nx,Ny)
            y=self.m1_CF.y.reshape(Nx,Ny)
            p1=axs.pcolor(x,y,
                          20*np.log10(np.abs(F_M1_CF.reshape(Ny,Nx))),
                          cmap=cmap,vmin=vmin,vmax=vmax)

            axs.axis('equal')
            #axs[1].axis('equal')
            plt.show()

            # Field on IF
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            cmap='jet'
            x=self.fimag2.x.reshape(int(self.fimag2_N[1]),int(self.fimag2_N[0]))
            y=self.fimag2.y.reshape(int(self.fimag2_N[1]),int(self.fimag2_N[0]))
            F_IF=F_IF.reshape(int(self.fimag2_N[1]),int(self.fimag2_N[0]))
            p1=axs[0].pcolor(x[0,:],y[:,0],20*np.log10(np.abs(F_IF)),cmap='jet')
            axs[0].axis('equal')
            p2=axs[1].pcolor(x,y,np.angle(F_IF)*180/np.pi,cmap='jet',vmax=180,vmin=-180)
            axs[1].axis('equal')
            plt.show()

            # Field on M2
            fig, axs = plt.subplots(1, 1, figsize=(12, 5))
            cmap='jet'
            M2_panelN=int(self.Panel_center_M2.size/2)
            Nx=self.M2_N[0]
            Ny=self.M2_N[1]
            N=Nx*Ny
            vmax=20*np.log10(np.abs(F_M2).max())
            vmin=vmax-15
            for n in range(M2_panelN):
                x=self.m2_0.x[n*N:(n+1)*N].reshape(Ny,Nx)
                y=self.m2_0.y[n*N:(n+1)*N].reshape(Ny,Nx)
                p1=axs.pcolor(x,y,
                                 20*np.log10(np.abs(F_M2[n*N:(n+1)*N].reshape(Ny,Nx))),
                                 cmap=cmap,vmin=vmin,vmax=vmax)
            Nx=self.CF2_N[0]
            Ny=self.CF2_N[1]
            x=self.m2_CF.x.reshape(Nx,Ny)
            y=self.m2_CF.y.reshape(Nx,Ny)
            p1=axs.pcolor(x,y,
                          20*np.log10(np.abs(F_M2_CF.reshape(Ny,Nx))),
                          cmap=cmap,vmin=vmin,vmax=vmax)
            axs.axis('equal')
            plt.show()