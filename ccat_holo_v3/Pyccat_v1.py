import numpy as np
import time
import h5py
import matplotlib.pyplot as plt

from Kirchhoffpy.mirrorpy import profile,squarepanel,deformation,ImagPlane,adjuster
# 'profile' with 2d poynomial coefficients produces surface function f(x,y).
# 'squarepanel' is the rim of the panels of mirrors and sampling range. 
# 'deformation' is the panel surface changes caused by errors of panel adjusters.
# 'ImagPlane' is used to define the sampling range of IF field plane.
from Kirchhoffpy.coordinate_operations import Coord
from Kirchhoffpy.coordinate_operations import Transform_local2global as local2global
from Kirchhoffpy.coordinate_operations import Transform_global2local as global2local
from Kirchhoffpy.coordinate_operations import cartesian_to_spherical as cart2spher
# geometry models are smampled by discrete points. Coordinates of the points 
# are expressed by Coord.x, Coord.y, Coord.z
# other functions are known from theirs names used for transformations between coordinate
# systems by giving their origin point displacement and rotation angles.
from Kirchhoffpy.Kirchhoff import Complex, PO_scalar
# PO_scalar is the field solver. Compex is used to express complex fields on mirror surfaces
# and desired field region.
from Kirchhoffpy.Feedpy import Gaussibeam
# Input field from a Gaussian feed horn.

from Kirchhoffpy.zernike_torch import mkCFn as make_zernike;
from Kirchhoffpy.zernike_torch import N as poly_N;
# surface errors expressed by zernike polynomials.
c=299792458*1000

class CCAT_holo():
    def __init__(self,Model_folder,output_folder):
        self.output_folder=output_folder
        '''Geometrical parameters'''
        # define surface profile of M1 and M2 in their local coordinates
        M2_poly_coeff=np.genfromtxt(Model_folder+'/coeffi_m2.txt',delimiter=',')
        M1_poly_coeff=np.genfromtxt(Model_folder+'/coeffi_m1.txt',delimiter=',')
        R2=3000 # normlization factor
        R1=3000
        # the 2D polynomial surface  
        self.surface_m2=profile(M2_poly_coeff,R2)
        self.surface_m1=profile(M1_poly_coeff,R1)

        # panel size, center, and number of sampling points on each panel
        self.Panel_center_M2=np.genfromtxt(Model_folder+'/L_m2.txt',delimiter=',')
        self.Panel_center_M1=np.genfromtxt(Model_folder+'/L_m1.txt',delimiter=',')
        parameters=np.genfromtxt(Model_folder+'/input.txt',delimiter=',')[:,1]

        self.M2_size=parameters[0:2]
        self.M1_size=parameters[2:4]
        self.M2_N=parameters[10:12].astype(int)
        self.M1_N=parameters[12:14].astype(int)

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
        # panel adjuster position
        self.p_m2=parameters[6]
        self.q_m2=parameters[7]
        self.p_m1=parameters[8]
        self.q_m1=parameters[9]

        self.S_m2_xy=None
        self.S_m1_xy=None
        
        # Intermedia plane
        self.fimag_N=parameters[14:16]
        self.fimag_size=parameters[16:18]
        self.fimag,self.fimag_n,self.fimag_dA=ImagPlane(self.fimag_size[0],self.fimag_size[1],
                                                        self.fimag_N[0],self.fimag_N[1]  
                                                        )
        '''Electrical parameters'''
        electro_params=np.genfromtxt(Model_folder+'/electrical_parameters.txt',delimiter=',')[...,1]
        self.freq=electro_params[0]*10**9
        self.edge_taper=electro_params[1]
        self.Angle_taper=electro_params[2]/180*np.pi
        self.Lambda=c/self.freq
        self.k=2*np.pi/self.Lambda

        self.coords(defocus=[0,0,0])


    def coords(self,defocus=[0,0,0]):
        '''coordinates systems'''
        '''
        #angle# is angle change of local coordinates and global coordinates;
        #D#     is the distance between origin of local coord and global coord in global coordinates;
        '''
        '''
        some germetrical parametrs
        '''
        Theta_0  =  0.927295218001612; # offset angle of MR;
        Ls       =  12000.0;           # distance between focal point and SR
        Lm       =  6000.0;            # distance between MR and SR;
        L_fimag  = 18000+Ls
        F        = 20000               # equivalent focal length of M2

        self.angle_m2=[-(np.pi/2+Theta_0)/2,0,0] #  1. m2 and global co-ordinates
        self.D_m2=[0,-Lm*np.sin(Theta_0),0]
        
        self.angle_m1=[-Theta_0/2,0,0]          #  2. m1 and global co-ordinates
        self.D_m1=[0,0,Lm*np.cos(Theta_0)]
        
        self.angle_s=[0,np.pi,0];               #  3. source and global co-ordinates
        self.D_s=[0,0,0]
        
        self.angle_fimag=[-Theta_0,0,0];        #  4. fimag and global co-ordinates
        defocus_fimag=[0,0,0]
        defocus_fimag[2]=1/(1/F-1/(Ls+defocus[2]))+L_fimag
        defocus_fimag[1]=(F+L_fimag-defocus_fimag[2])/F*defocus[1]
        defocus_fimag[0]=(F+L_fimag-defocus_fimag[2])/F*defocus[0]
        self.D_fimag=[0,0,0]
        self.D_fimag[0]=defocus_fimag[0]
        self.D_fimag[1]=defocus_fimag[1]*np.cos(Theta_0)\
            -np.sin(Theta_0)*(L_fimag-defocus_fimag[2]+Lm)
        self.D_fimag[2]=-defocus_fimag[1]*np.sin(Theta_0)\
            -np.cos(Theta_0)*(L_fimag-defocus_fimag[2])
        
        # feed coordinate system
        '''
        C=1/(1/Lm-1/F)+defocus[2]+Ls;
        C=21000;
        angle_f=[np.pi/2-defocus[1]/C,0,-defocus[0]/C]; 
        D_f=[defocus[0],Ls+defocus[2]-Lm*np.sin(Theta_0),-defocus[1]];
        '''
        self.angle_f=[np.pi/2,0,0];    
        self.D_f=[defocus[0],Ls+defocus[2]-Lm*np.sin(Theta_0),-defocus[1]] 

    def beam(self,scan_file,defocus=[0,0,0],Feed_beam=Gaussibeam,Matrix=False):
        trace=np.genfromtxt(scan_file,delimiter=',')
        scan_pattern=Coord()
        scan_pattern.x=trace[:,0]
        scan_pattern.y=trace[:,1]
        scan_pattern.z=trace[:,2]

        ''' first beam pattern calculation'''
        filename=self.output_folder+'/data_Rx_dx'+str(defocus[0])+'_dy'+str(defocus[1])+'_dz'+str(defocus[2])+'.h5py'
        # Set receiver location
        self.coords(defocus=defocus)
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
        start=time.perf_counter()
        # 1. convert MIRROR 2 into global coordinate system
        m2=local2global(self.angle_m2,self.D_m2,self.m2)
        m2_n=local2global(self.angle_m2,[0,0,0],self.m2_n)
        Field_m2=Complex()

        # 2. illumination field on M2
        Field_m2.real,Field_m2.imag,cosm2_i=Feed_beam(self.edge_taper,
                                                      self.Angle_taper,
                                                      self.k,
                                                      m2,m2_n,
                                                      self.angle_f,self.D_f
                                                      )

        # 3. calculate field on IF plane
        fimag=local2global(self.angle_fimag,self.D_fimag,self.fimag)
        fimag_n=local2global(self.angle_fimag,[0,0,0],self.fimag_n)
        Matrix1,Field_fimag,cosm2_r=PO_scalar(m2,m2_n,
                                              self.m2_dA,
                                              fimag,cosm2_i,
                                              Field_m2,-self.k,
                                              Keepmatrix=Matrix
                                              )
        
        # 4. calculate field on M1
        m1=local2global(self.angle_m1,self.D_m1,self.m1)
        m1_n=local2global(self.angle_m1,[0,0,0],self.m1_n)
        self.aperture_xy=np.append(m1.x,m1.y).reshape(2,-1)

        NN=int(fimag.x.size/2)
        Fimag0=[fimag.x[NN],fimag.y[NN],fimag.z[NN]]
        x=Fimag0[0].item()-m1.x.reshape(1,-1)
        y=Fimag0[1].item()-m1.y.reshape(1,-1)
        z=Fimag0[2].item()-m1.z.reshape(1,-1)
        
        r=np.sqrt(x**2+y**2+z**2)
        cosm1_i=(x*m1_n.x+y*m1_n.y+z*m1_n.z)/r
        #cosm1_i=T.tensor(cosm1_i).to(DEVICE);
        del(x,y,z,r)

        Matrix2,Field_m1,cosm=PO_scalar(fimag,fimag_n,self.fimag_dA,
                                        m1,np.array([1]),
                                        Field_fimag,
                                        self.k,
                                        Keepmatrix=Matrix
                                        )
        del(cosm)
        
        #5. calculate the field in the source range;
        source=local2global(self.angle_s,self.D_s,scan_pattern)
        Matrix3,Field_s,cosm1_r=PO_scalar(m1,m1_n,self.m1_dA,
                                          source,cosm1_i,
                                          Field_m1,
                                          self.k,
                                          Keepmatrix=Matrix
                                          )
        
        '''
        emerging m1 and m2 to m12
        '''    
        Matrix21=Complex()
        if Matrix:
            Matrix21.real=np.matmul(Matrix2.real,Matrix1.real)-np.matmul(Matrix2.imag,Matrix1.imag)
            Matrix21.imag=np.matmul(Matrix2.real,Matrix1.imag)+np.matmul(Matrix2.imag,Matrix1.real)
        else:
            pass
    
        elapsed =(time.perf_counter()-start)
        print('time used:',elapsed)
        with h5py.File(filename,'w') as f:
              f.create_dataset('M21_real',data=Matrix21.real)
              f.create_dataset('M21_imag',data=Matrix21.imag)
              f.create_dataset('M3_real',data=Matrix3.real)
              f.create_dataset('M3_imag',data=Matrix3.imag)
              f.create_dataset('cosm2_i',data=cosm2_i)
              f.create_dataset('cosm2_r',data=cosm2_r)
              f.create_dataset('cosm1_i',data=cosm1_i)
              f.create_dataset('cosm1_r',data=cosm1_r)
              f.create_dataset('F_m2_real',data=Field_m2.real)
              f.create_dataset('F_m2_imag',data=Field_m2.imag)
              f.create_dataset('F_m1_real',data=Field_m1.real)
              f.create_dataset('F_m1_imag',data=Field_m1.imag)
              f.create_dataset('F_if_real',data=Field_fimag.real)
              f.create_dataset('F_if_imag',data=Field_fimag.imag)
              f.create_dataset('F_beam_real',data=Field_s.real)
              f.create_dataset('F_beam_imag',data=Field_s.real)
              f.create_dataset('aperture',data=self.aperture_xy)
              f.create_dataset('scan_pattern',data=np.concatenate((source.x,
                                                                   source.y,source.z)).reshape(3,-1))

    def plot_beam(self,defocus=[0,0,0]):
        filename=self.output_folder+'data_Rx_dx'+str(defocus[0])+'_dy'+str(defocus[1])+'_dz'+str(defocus[2])
        with h5py.File(filename,'r') as f:
            beam=f['F_beam_real'][:]+1j*f['F_beam_imag'][:]
            NN=int(np.sqrt(f['scan_pattern'][:].size))
            xyz=f['scan_pattern'][:]
            x=xyz[0,:].reshape(NN,-1)
            y=xyz[1,:].reshape(NN,-1)
            del(xyz)
        beam=beam.reshape(NN,-1)
        fig, axs = plt.subplots(1, 2, figsize=(9, 3))
        cmap='jet'
        p1=axs[0].pcolor(x,y,20*np.log10(beam),cmap='jet')
        p2=axs[1].pcolor(x,y,np.angle(beam)*180/np.pi,cmap='jet')
        fig.show()