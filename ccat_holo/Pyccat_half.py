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
        Ox1=self.M1_CF_4points[0,0]
        Oy1=self.M1_CF_4points[0,1]

        Ox2=self.M2_CF_4points[0,0]
        Oy2=self.M2_CF_4points[0,1]

        v11=self.M1_CF_4points[1,:]-self.M1_CF_4points[0,:]
        v12=self.M1_CF_4points[3,:]-self.M1_CF_4points[0,:]

        v21=self.M2_CF_4points[1,:]-self.M2_CF_4points[0,:]
        v22=self.M2_CF_4points[3,:]-self.M2_CF_4points[0,:]

        '''surface'''
        # define surface profile of M1 and M2 in their local coordinates
        CF2_poly_coeff=np.genfromtxt(Model_folder+'/coeffi_m2CF.txt',delimiter=',')
        CF1_poly_coeff=np.genfromtxt(Model_folder+'/coeffi_m1CF.txt',delimiter=',')
        # the 2D polynomial surface  
        self.surface_m2_CF=profile(CF2_poly_coeff,1)
        self.surface_m1_CF=profile(CF1_poly_coeff,1)
        """
        self.m2_CF,self.m2_CF_n,self.m2_CF_dA=parallelogram_panel(Ox1,Oy1,
                                                                  v11,v12,
                                                                  N11,N12,
                                                                  self.surface_m2_CF,
                                                                  quadrature='uniform')
        self.m1_CF,self.m1_CF_n,self.m1_CF_dA=parallelogram_panel(Ox2,Oy2,
                                                                  v21,v22,
                                                                  N21,N22,
                                                                  self.surface_m1_CF,
                                                                  quadrature='uniform')
        """
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

        self.widget.add_mesh(CF1,show_edges=True,color="gray")
        self.widget.add_mesh(CF2,show_edges=True,color="gray")

    def _beam(self,scan_file,Rx=[0,0,0],
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
        self._coords(Rx=Rx)

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

        NN=int(fimag.x.size/2)
        Fimag0=[fimag.x[NN],fimag.y[NN],fimag.z[NN]]
        x=Fimag0[0].item()-m1.x.reshape(1,-1)
        y=Fimag0[1].item()-m1.y.reshape(1,-1)
        z=Fimag0[2].item()-m1.z.reshape(1,-1)
        
        r=np.sqrt(x**2+y**2+z**2)
        cosm1_i=(x*m1_n.x+y*m1_n.y+z*m1_n.z)/r
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
        Matrix3,self.Field_s,cosm1_r=PO_scalar(m1,m1_n,self.m1_dA,
                                          source,cosm1_i,
                                          Field_m1,
                                          self.k,
                                          Keepmatrix=Matrix
                                          )
        
        elapsed =(time.perf_counter()-start)
        print('time used:',elapsed)
        # Save the computation data into h5py file, and the intermediate Matrixs that wil
        # be used for accelerating the forward beam calculations.
        self.output_filename=self.output_folder+'/'+file_name+'_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
        with h5py.File(self.output_filename,'w') as f:
              #f.create_dataset('conf',data=(Rx,scan_file))
              f.create_dataset('freq (GHz)',data=self.freq)
              f.create_dataset('M21_real',data=Matrix21.real)
              f.create_dataset('M21_imag',data=Matrix21.imag)
              f.create_dataset('M3_real',data=Matrix3.real)
              f.create_dataset('M3_imag',data=Matrix3.imag)
              del(Matrix21,Matrix3)
              f.create_dataset('cosm2_i',data=cosm2_i)
              f.create_dataset('cosm2_r',data=cosm2_r)
              f.create_dataset('cosm1_i',data=cosm1_i)
              f.create_dataset('cosm1_r',data=cosm1_r)
              f.create_dataset('F_m2_real',data=Field_m2.real)
              f.create_dataset('F_m2_imag',data=Field_m2.imag)
              f.create_dataset('F_m1_real',data=Field_m1.real)
              f.create_dataset('F_m1_imag',data=Field_m1.imag)
              f.create_dataset('F_if_real',data=self.Field_fimag.real)
              f.create_dataset('F_if_imag',data=self.Field_fimag.imag)
              f.create_dataset('F_beam_real',data=self.Field_s.real)
              f.create_dataset('F_beam_imag',data=self.Field_s.imag)
              f.create_dataset('aperture',data=self.aperture_xy)
              f.create_dataset('scan_pattern',data=np.concatenate((source.x,
                                                                   source.y,
                                                                   source.z)).reshape(3,-1))

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
                F_IF=f['F_if_real'][:]+1j*f['F_if_imag'][:]
                F_M2=f['F_m2_real'][:]+1j*f['F_m2_imag'][:]
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
            vmax=20*np.log10(np.abs(F_M1).max())
            vmin=vmax-20
            for n in range(M1_panelN):
                x=self.m1_0.x[n*N:(n+1)*N].reshape(Ny,Nx)
                y=self.m1_0.y[n*N:(n+1)*N].reshape(Ny,Nx)
                p1=axs.pcolor(x,y,
                                 20*np.log10(np.abs(F_M1[n*N:(n+1)*N].reshape(Ny,Nx))),
                                 cmap=cmap,vmin=vmin,vmax=vmax)
            axs.axis('equal')
            #axs[1].axis('equal')
            plt.show()

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
            # Field on M2
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
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
                p1=axs[0].pcolor(x,y,
                                 20*np.log10(np.abs(F_M2[n*N:(n+1)*N].reshape(Ny,Nx))),
                                 cmap=cmap,vmin=vmin,vmax=vmax)
                p2=axs[1].pcolor(x,y,
                                 np.angle(F_M2[n*N:(n+1)*N].reshape(Ny,Nx))*180/np.pi,
                                 cmap=cmap,vmin=-180,vmax=180)
            axs[0].axis('equal')
            axs[1].axis('equal')
            plt.show()