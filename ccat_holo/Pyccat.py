import os

import numpy as np
import scipy.optimize
import torch as T
import time
import h5py

from matplotlib import cm
import matplotlib.pyplot as plt
import pyvista as pv
pv.set_jupyter_backend('trame')#('static')#


from mirrorpy import profile,squarepanel,deformation,ImagPlane,adjuster
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
from Kirchhoff import Complex, PO_scalar
# PO_scalar is the field solver. Compex is used to express complex fields on mirror surfaces
# and desired field region.
from Feedpy import Gaussibeam
# Input field from a Gaussian feed horn.

from zernike_torch import mkCFn as make_zernike;
from zernike_torch import N as poly_N;

from inference import DATA2TORCH, correctphase2
from inference import fitting_func, fitting_func_zernike

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

def Load_Mat(filename):
    with h5py.File(filename,'r') as f:
        M21=Complex()
        M21.real=f['M21_real'][:]
        M21.imag=f['M21_imag'][:]
        M3=Complex()
        M3.real=f['M3_real'][:]
        M3.imag=f['M3_imag'][:]
        cosm2_i=f['cosm2_i'][:]
        cosm2_r=f['cosm2_r'][:]
        cosm1_i=f['cosm1_i'][:]
        cosm1_r=f['cosm1_r'][:]
        F_m2=Complex()
        F_m2.real=f['F_m2_real'][:]
        F_m2.imag=f['F_m2_imag'][:]
    return M21,M3,cosm2_i,cosm2_r,cosm1_i,cosm1_r,F_m2
class CCAT_holo():
    def __init__(self,Model_folder,output_folder,holo_conf=holo_setup,input_Rx_beam=Gaussibeam):
        if os.path.isdir(output_folder):
            pass
        else:
            os.makedirs(output_folder)
        self.Model_folder=Model_folder
        self.output_folder=output_folder
        self.holo_conf=holo_conf
        self.output_filename=None
        self.View_3D=None
        self.Rx_3D=dict.fromkeys(holo_conf.keys(),[])
        ### configure the 3D view widget
        self.widget=pv.Plotter(notebook=True)
        _ = self.widget.add_axes(
            line_width=5,
            cone_radius=0.6,
            shaft_length=0.7,
            tip_length=0.3,
            ambient=0.5,
            label_size=(0.4, 0.16),
        )
        _ = self.widget.add_bounding_box(line_width=5, color='black')

        '''Geometrical parameters'''
        # define surface profile of M1 and M2 in their local coordinates
        M2_poly_coeff=np.genfromtxt(Model_folder+'/coeffi_m2.txt',delimiter=',')
        M1_poly_coeff=np.genfromtxt(Model_folder+'/coeffi_m1.txt',delimiter=',')
        self.R2=3000 # normlization factor
        self.R1=3000
        # the 2D polynomial surface  
        self.surface_m2=profile(M2_poly_coeff,self.R2)
        self.surface_m1=profile(M1_poly_coeff,self.R1)

        # panel size, center, and number of sampling points on each panel
        self.Panel_center_M2=np.genfromtxt(Model_folder+'/L_m2.txt',delimiter=',')
        self.Panel_center_M1=np.genfromtxt(Model_folder+'/L_m1.txt',delimiter=',')
        parameters=np.genfromtxt(Model_folder+'/Model.txt',delimiter=',')[:,1]

        self.M2_size=parameters[0:2]
        self.M1_size=parameters[2:4]
        self.M2_N=parameters[10:12].astype(int)
        self.M1_N=parameters[12:14].astype(int)

        self.m2_0,self.m2_n,self.m2_dA=squarepanel(self.Panel_center_M2[...,0],self.Panel_center_M2[...,1],
                                                    self.M2_size[0],self.M2_size[1],
                                                    self.M2_N[0],self.M2_N[1],
                                                    self.surface_m2
                                                    )
        self.m1_0,self.m1_n,self.m1_dA=squarepanel(self.Panel_center_M1[...,0],self.Panel_center_M1[...,1],
                                                    self.M1_size[0],self.M1_size[1],
                                                    self.M1_N[0],self.M1_N[1],
                                                    self.surface_m1
                                                    )
        # panel adjuster position
        self.p_m2=parameters[6]
        self.q_m2=parameters[7]
        self.p_m1=parameters[8]
        self.q_m1=parameters[9]

        self.S2_x,self.S2_y,self.S1_x,self.S1_y=adjuster(self.Panel_center_M2,self.Panel_center_M1,
                                                         self.p_m2,self.q_m2,
                                                         self.p_m1,self.q_m1,
                                                         self.R2,self.R1
                                                         )
        
        # Intermedia plane
        self.fimag_N=parameters[14:16]
        self.fimag_size=parameters[16:]
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
        # beam profile of the feedhorn of the receiver
        self.input_feed_beam=input_Rx_beam
        
        self._coords(Rx=[0,0,0])

        self.FF=None
        print('FYST telescope model has been created!!')

    def _coords(self,Rx=[0,0,0]):
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
        
        # feed coordinate system
        '''
        C=1/(1/Lm-1/F)+defocus[2]+Ls;
        C=21000;
        angle_f=[np.pi/2-defocus[1]/C,0,-defocus[0]/C]; 
        D_f=[defocus[0],Ls+defocus[2]-Lm*np.sin(Theta_0),-defocus[1]];
        '''
        self.angle_f=[np.pi/2,0,0];    
        self.D_f=[Rx[0],Ls+Rx[2]-Lm*np.sin(Theta_0),-Rx[1]] 
    def view(self):
        '''show the telescope model'''
        self.widget.clear()
        m2=Coord()
        m1=Coord()
        N_m2=self.Panel_center_M2[...,0].size
        N_m1=self.Panel_center_M1[...,0].size
        a2=self.M2_size[0]
        b2=self.M2_size[1]
        a1=self.M1_size[0]
        b1=self.M1_size[1]
        m2.x=self.Panel_center_M2[:,0].repeat(4).reshape(-1,4)
        m2.y=self.Panel_center_M2[:,1].repeat(4).reshape(-1,4)
        ###########
        m2.x[:,0]=m2.x[:,0]-a2/2; m2.y[:,0]=m2.y[:,0]-b2/2
        m2.x[:,1]=m2.x[:,1]-a2/2; m2.y[:,1]=m2.y[:,1]+b2/2
        m2.x[:,2]=m2.x[:,2]+a2/2; m2.y[:,2]=m2.y[:,2]+b2/2
        m2.x[:,3]=m2.x[:,3]+a2/2; m2.y[:,3]=m2.y[:,3]-b2/2
        m2.x=m2.x.ravel()
        m2.y=m2.y.ravel()
        m2.z,V_n=self.surface_m2(m2.x,m2.y)
        del(V_n)


        m1.x=self.Panel_center_M1[...,0].repeat(4).reshape(-1,4)
        m1.y=self.Panel_center_M1[...,1].repeat(4).reshape(-1,4)
        m1.x[:,0]=m1.x[:,0]-a1/2; m1.y[:,0]=m1.y[:,0]-b1/2
        m1.x[:,1]=m1.x[:,1]-a1/2; m1.y[:,1]=m1.y[:,1]+b1/2
        m1.x[:,2]=m1.x[:,2]+a1/2; m1.y[:,2]=m1.y[:,2]+b1/2
        m1.x[:,3]=m1.x[:,3]+a1/2; m1.y[:,3]=m1.y[:,3]-b1/2

        m1.x=m1.x.ravel()
        m1.y=m1.y.ravel()

        m1.z,V_n=self.surface_m1(m1.y,m1.y)

        print(self.angle_m2,self.D_m2)
        m2=local2global(self.angle_m2,self.D_m2,m2)
        m1=local2global(self.angle_m1,self.D_m1,m1)

        points2 = np.c_[m2.x.reshape(-1), m2.y.reshape(-1), m2.z.reshape(-1)]
        points1 = np.c_[m1.x.reshape(-1), m1.y.reshape(-1), m1.z.reshape(-1)]
        del(m2,m1)

        faces2=np.ones(N_m2).astype(int)*4
        factor=np.linspace(0,4*N_m2-1,4*N_m2).astype(int).reshape(-1,4)
        faces2=np.c_[faces2,factor].ravel()


        faces1=np.ones(N_m1).astype(int)*4
        factor=np.linspace(0,4*N_m1-1,4*N_m1).astype(int).reshape(-1,4)
        faces1=np.c_[faces1,factor]

        
        panel1 = pv.PolyData(points1,faces1)
        panel2 = pv.PolyData(points2,faces2)

        self.widget.add_mesh(panel1,show_edges=True)
        self.widget.add_mesh(panel2,show_edges=True)
        self.view_Rx()
        self.widget.show()

    def view_Rx(self,Rx=[]):
        '''highlight the required Receiver. '''
        for key in self.holo_conf:
            self._coords(self.holo_conf[key][0])
            cone=pv.Cone(center=tuple(self.D_f),
                        direction=(0,1,0),
                        height=300*self.Lambda,
                        radius=100*self.Lambda)
            if self.Rx_3D[key]!=[]:
                self.widget.remove_actor(self.Rx_3D[key])
            if key in Rx:
                self.Rx_3D[key]=self.widget.add_mesh(cone,color='blue')
            else:
                self.Rx_3D[key]=self.widget.add_mesh(cone)


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
        Field_m2.real,Field_m2.imag,cosm2_i=Feed_beam(self.edge_taper,
                                                      self.Angle_taper,
                                                      self.k,
                                                      m2,m2_n,
                                                      self.angle_f,self.D_f
                                                      )

        # 3. calculate field on IF plane
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
                                                                   source.y,source.z)).reshape(3,-1))

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
            with h5py.File(filename,'r') as f:
                beam=f['F_beam_real'][:]+1j*f['F_beam_imag'][:]
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

            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            cmap='jet'
            p1=axs[0].plot(x[50,:],20*np.log10(np.abs(beam[50,:])))
            p2=axs[1].plot(x[50,:],np.angle(beam[50,:])*180/np.pi,'*-')
            plt.show()
            #print(beam.real)
            #print(beam.imag)

    def First_Beam_cal(self,S2_init=np.zeros((5,69)),S1_init=np.zeros((5,77)),Error_m2=0,Error_m1=0,Matrix=True):
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
                self._beam(self.holo_conf[keys][1],Rx=self.holo_conf[keys][0],Matrix=Matrix,S2_init=S2_init,S1_init=S1_init,Error_m2=Error_m2,Error_m1=Error_m1)
    
    def mk_FF(self,fitting_param='panel adjusters',
                         Device=T.device('cpu'),
                         Z_order=7,
                         Memory_reduc=False):
        '''Define forward function (FF) !
           *** fitting_param ***  1. 'panel adjusters' the fitting parameters are adjuster movements.
                                   2. 'zernike' fit the coefficients of a set of zernike polynomials
                                      for large spatial scale surface errors.
                                      'zernike' model is chosen, maximum zernike order 'Z_order' 
                                      must be given.
        '''
        if T.cuda.is_available():
            T.cuda.empty_cache()
        if self.FF!=None:
            del(self.FF)
        # read the aperture field points coordinates
        Rx=self.holo_conf['Rx1'][0]
        file=self.output_folder+'/data_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
        with h5py.File(file,'r') as f:
            self.aperture_xy=f['aperture'][:][:]
        Aperture=DATA2TORCH(self.aperture_xy,DEVICE=Device)[0]
        List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1=DATA2TORCH(self.Panel_center_M2,self.Panel_center_M1,
                                                             self.m2_0,self.m1_0,
                                                             self.p_m2,self.q_m2,
                                                             self.p_m1,self.q_m1,DEVICE=Device)
        Vars={'Rx1': None, 'Rx2': None, 'Rx3': None, 'Rx4': None, 'Rx5': None}
        N=len(Vars)
        if Memory_reduc:
            for i in range(3):
                Rx=self.holo_conf['Rx'+str(i+1)][0]
                file=self.output_folder+'/data_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
                Vars['Rx'+str(i+1)]=DATA2TORCH(*Load_Mat(file),DEVICE=Device)
            flip_order2=[]
            N_m2=self.Panel_center_M2.shape[0]
            for i in range(self.Panel_center_M2.shape[0]):
                N1=np.where(self.Panel_center_M2[:,0]==-self.Panel_center_M2[i,0])[0]
                N2=np.where(self.Panel_center_M2[:,1]==self.Panel_center_M2[i,1])[0]
                NN=np.intersect1d(N1,N2)
                flip_order2.append(NN.tolist())
            flip_order1=[]
            N_m1=self.Panel_center_M1.shape[0]
            for i in range(self.Panel_center_M1.shape[0]):
                N1=np.where(self.Panel_center_M1[:,0]==-self.Panel_center_M1[i,0])[0]
                N2=np.where(self.Panel_center_M1[:,1]==self.Panel_center_M1[i,1])[0]
                NN=np.intersect1d(N1,N2)
                flip_order1.append(NN.tolist())
        else:
            for i in range(N):
                Rx=self.holo_conf['Rx'+str(i+1)][0]
                file=self.output_folder+'/data_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
                Vars['Rx'+str(i+1)]=DATA2TORCH(*Load_Mat(file),DEVICE=Device)
            
        if fitting_param=='panel adjusters':
            if Memory_reduc:
                def FF(Adjusters,Para_A,Para_p):
                    AD2=Adjusters[0:5*N_m2].reshape(5,-1)[:,flip_order2].view(-1)
                    AD1=Adjusters[0:5*N_m1].reshape(5,-1)[:,flip_order1].view(-1)
                    Adjustersp=T.cat((AD2,AD1))
                    R0=fitting_func(*Vars['Rx1'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[0:6],Para_p[0:5],Aperture)
                    R1=fitting_func(*Vars['Rx2'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6:6*2],Para_p[5:5*2],Aperture)
                    R2=fitting_func(*Vars['Rx3'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6*2:6*3],Para_p[5*2:5*3],Aperture)
                    R3=fitting_func(*Vars['Rx2'],
                                    Adjustersp,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6*3:6*4],Para_p[5*3:5*4],Aperture)
                    R4=fitting_func(*Vars['Rx3'],
                                    Adjustersp,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6*4:],Para_p[5*4:],Aperture)
                    return T.cat((R0,R1,R2,R3,R4))
            else:
                def FF(Adjusters,Para_A,Para_p):
                    
                    R0=fitting_func(*Vars['Rx1'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[0:6],Para_p[0:5],Aperture)
                    R1=fitting_func(*Vars['Rx2'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6:6*2],Para_p[5:5*2],Aperture)
                    R2=fitting_func(*Vars['Rx3'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6*2:6*3],Para_p[5*2:5*3],Aperture)
                    R3=fitting_func(*Vars['Rx4'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6*3:6*4],Para_p[5*3:5*4],Aperture)
                    R4=fitting_func(*Vars['Rx5'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6*4:],Para_p[5*4:],Aperture)
                    return T.cat((R0,R1,R2,R3,R4))
            self.FF=FF
        elif fitting_param=='zernike':
            self.Z_surf2=make_zernike(Z_order,m2.x/self.R2,(m2.y+0.0)/self.R2,dtype='torch',device=Device)
            self.Z_surf1=make_zernike(Z_order,m1.x/self.R1,(m1.y-0.0)/self.R1,dtype='torch',device=Device)
            self.Z_order=Z_order
            if Memory_reduc:
                pass
            else:
                def FF(Z_coeff,Para_A,Para_p):
                    
                    R0=fitting_func_zernike(*Vars['Rx1'],
                                            Z_coeff[0,:],Z_coeff[1,:],self.Z_surf2,self.Z_surf1,
                                            self.k,
                                            Para_A[0:6],Para_p[0:5],Aperture)
                    R1=fitting_func_zernike(*Vars['Rx2'],
                                            Z_coeff[0,:],Z_coeff[1,:],self.Z_surf2,self.Z_surf1,
                                            self.k,
                                            Para_A[6:6*2],Para_p[5:5*2],Aperture)
                    R2=fitting_func_zernike(*Vars['Rx3'],
                                            Z_coeff[0,:],Z_coeff[1,:],self.Z_surf2,self.Z_surf1,
                                            self.k,
                                            Para_A[6*2:6*3],Para_p[5*2:5*3],Aperture)
                    R3=fitting_func_zernike(*Vars['Rx4'],
                                            Z_coeff[0,:],Z_coeff[1,:],self.Z_surf2,self.Z_surf1,
                                            self.k,
                                            Para_A[6*3:6*4],Para_p[5*3:5*4],Aperture)
                    R4=fitting_func_zernike(*Vars['Rx5'],
                                            Z_coeff[0,:],Z_coeff[1,:],self.Z_surf2,self.Z_surf1,
                                            self.k,
                                            Para_A[6*4:],Para_p[5*4:],Aperture)
                    return T.cat((R0,R1,R2,R3,R4))
            self.FF=FF
        else:
            print('The forward function is not sucessfully created!!!!')
    
    def mk_FF_4maps(self,fitting_param='panel adjusters',
                         Device=T.device('cpu'),
                         Z_order=7,
                         Memory_reduc=False):
        '''Define forward function (FF) !
           *** fitting_param ***  1. 'panel adjusters' the fitting parameters are adjuster movements.
                                   2. 'zernike' fit the coefficients of a set of zernike polynomials
                                      for large spatial scale surface errors.
                                      'zernike' model is chosen, maximum zernike order 'Z_order' 
                                      must be given.
        '''
        if T.cuda.is_available():
            T.cuda.empty_cache()
        if self.FF!=None:
            del(self.FF)
        # read the aperture field points coordinates
        Rx=self.holo_conf['Rx2'][0]
        file=self.output_folder+'/data_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
        with h5py.File(file,'r') as f:
            self.aperture_xy=f['aperture'][:][:]
        Aperture=DATA2TORCH(self.aperture_xy,DEVICE=Device)[0]
        List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1=DATA2TORCH(self.Panel_center_M2,self.Panel_center_M1,
                                                             self.m2_0,self.m1_0,
                                                             self.p_m2,self.q_m2,
                                                             self.p_m1,self.q_m1,DEVICE=Device)
        Vars={'Rx2': None, 'Rx3': None, 'Rx4': None, 'Rx5': None}
        N=len(Vars)
        if Memory_reduc:
            for i in range(2):
                Rx=self.holo_conf['Rx'+str(i+2)][0]
                file=self.output_folder+'/data_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
                Vars['Rx'+str(i+2)]=DATA2TORCH(*Load_Mat(file),DEVICE=Device)
            flip_order2=[]
            N_m2=self.Panel_center_M2.shape[0]
            for i in range(self.Panel_center_M2.shape[0]):
                N1=np.where(self.Panel_center_M2[:,0]==-self.Panel_center_M2[i,0])[0]
                N2=np.where(self.Panel_center_M2[:,1]==self.Panel_center_M2[i,1])[0]
                NN=np.intersect1d(N1,N2)
                flip_order2.append(NN.tolist())
            flip_order1=[]
            N_m1=self.Panel_center_M1.shape[0]
            for i in range(self.Panel_center_M1.shape[0]):
                N1=np.where(self.Panel_center_M1[:,0]==-self.Panel_center_M1[i,0])[0]
                N2=np.where(self.Panel_center_M1[:,1]==self.Panel_center_M1[i,1])[0]
                NN=np.intersect1d(N1,N2)
                flip_order1.append(NN.tolist())
        else:
            for i in range(N):
                Rx=self.holo_conf['Rx'+str(i+2)][0]
                file=self.output_folder+'/data_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
                Vars['Rx'+str(i+2)]=DATA2TORCH(*Load_Mat(file),DEVICE=Device)
        
        if fitting_param=='panel adjusters':
            if Memory_reduc:
                def FF(Adjusters,Para_A,Para_p):
                    AD2=Adjusters[0:5*N_m2].reshape(5,-1)[:,flip_order2].view(-1)
                    AD1=Adjusters[0:5*N_m1].reshape(5,-1)[:,flip_order1].view(-1)
                    Adjustersp=T.cat((AD2,AD1)).reshape(-1)
                    R1=fitting_func(*Vars['Rx2'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[0:6*1],Para_p[0:5*1],Aperture)
                    R2=fitting_func(*Vars['Rx3'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6:6*2],Para_p[5:5*2],Aperture)
                    R3=fitting_func(*Vars['Rx2'],
                                    Adjustersp,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6*2:6*3],Para_p[5*2:5*3],Aperture)
                    R4=fitting_func(*Vars['Rx3'],
                                    Adjustersp,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6*3:],Para_p[5*3:],Aperture)
                    return T.cat((R1,R2,R3,R4))
            else:
                def FF(Adjusters,Para_A,Para_p):
                    R1=fitting_func(*Vars['Rx2'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[0:6*1],Para_p[0:5*1],Aperture)
                    R2=fitting_func(*Vars['Rx3'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6:6*2],Para_p[5:5*2],Aperture)
                    R3=fitting_func(*Vars['Rx4'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6*2:6*3],Para_p[5*2:5*3],Aperture)
                    R4=fitting_func(*Vars['Rx5'],
                                    Adjusters,
                                    List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                    Para_A[6*3:],Para_p[5*3:],Aperture)
                    return T.cat((R1,R2,R3,R4))
            self.FF=FF
        elif fitting_param=='zernike':
            self.Z_surf2=make_zernike(Z_order,m2.x/self.R2,(m2.y+0.0)/self.R2,dtype='torch',device=Device)
            self.Z_surf1=make_zernike(Z_order,m1.x/self.R1,(m1.y-0.0)/self.R1,dtype='torch',device=Device)
            self.Z_order=Z_order
            def FF(Z_coeff,Para_A,Para_p):
                Z_coeff=Z_coeff.reshape(2,-1)
                R1=fitting_func_zernike(*Vars['Rx2'],
                                        Z_coeff[0,:],Z_coeff[1,:],self.Z_surf2,self.Z_surf1,
                                        self.k,
                                        Para_A[0:6*1],Para_p[0:5*1],Aperture)
                R2=fitting_func_zernike(*Vars['Rx3'],
                                        Z_coeff[0,:],Z_coeff[1,:],self.Z_surf2,self.Z_surf1,
                                        self.k,
                                        Para_A[6:6*2],Para_p[5:5*2],Aperture)
                R3=fitting_func_zernike(*Vars['Rx4'],
                                        Z_coeff[0,:],Z_coeff[1,:],self.Z_surf2,self.Z_surf1,
                                        self.k,
                                        Para_A[6*2:6*3],Para_p[5*2:5*3],Aperture)
                R4=fitting_func_zernike(*Vars['Rx5'],
                                        Z_coeff[0,:],Z_coeff[1,:],self.Z_surf2,self.Z_surf1,
                                        self.k,
                                        Para_A[6*3:],Para_p[5*3:],Aperture)
                return T.cat((R1,R2,R3,R4))
            self.FF=FF
        else:
            print('The forward function is not sucessfully created!!!!')

    def mk_FF_1map(self,fitting_param='panel adjusters',
                        Device=T.device('cpu'),
                        Z_order=7,
                        conf='Rx1'):
        '''Define forward function (FF) for one-beam holography analysis!
           *** fitting_param ***  1. 'panel adjusters' the fitting parameters are adjuster movements.
                                   2. 'zernike' fit the coefficients of a set of zernike polynomials
                                      for large spatial scale surface errors.
                                      'zernike' model is chosen, maximum zernike order 'Z_order' 
                                      must be given.
        '''
        if T.cuda.is_available():
            T.cuda.empty_cache()
        if self.FF!=None:
            del(self.FF)
        # read the aperture field points coordinates
        Rx=self.holo_conf[conf][0]
        file=self.output_folder+'/data_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
        with h5py.File(file,'r') as f:
            self.aperture_xy=f['aperture'][:][:]
        Aperture=DATA2TORCH(self.aperture_xy,DEVICE=Device)[0]
        List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1=DATA2TORCH(self.Panel_center_M2,self.Panel_center_M1,
                                                             self.m2_0,self.m1_0,
                                                             self.p_m2,self.q_m2,
                                                             self.p_m1,self.q_m1,DEVICE=Device)
        Vars={conf: None}
        file=self.output_folder+'/data_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
        Vars[conf]=DATA2TORCH(*Load_Mat(file),DEVICE=Device)
        
        if fitting_param=='panel adjusters':
            def FF(Adjusters,Para_A,Para_p):
                R0=fitting_func(*Vars[conf],
                                Adjusters,
                                List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,
                                Para_A[0:6],Para_p[0:5],Aperture)
                return R0
            self.FF=FF
        elif fitting_param=='zernike':
            self.Z_surf2=make_zernike(Z_order,m2.x/self.R2,(m2.y+0.0)/self.R2,dtype='torch',device=Device)
            self.Z_surf1=make_zernike(Z_order,m1.x/self.R1,(m1.y-0.0)/self.R1,dtype='torch',device=Device)
            self.Z_order=Z_order
            def FF(Z_coeff,Para_A,Para_p):
                Z_coeff=Z_coeff.reshape(2,-1)
                R0=fitting_func_zernike(*Vars[conf],
                                        Z_coeff[0,:],Z_coeff[1,:],self.Z_surf2,self.Z_surf1,
                                        self.k,
                                        Para_A[0:6],Para_p[0:5],Aperture)
                return R0
            self.FF=FF
        else:
            print('The forward function is not sucessfully created!!!!')
    
    
    '''
    methods for holographic analysis
    '''
    
    def fit_LP(self,
               Meas_maps,
               Device=T.device('cpu'),
               Init=np.zeros((5*(69+77))),
               Map_N=5,
               outputfilename='fit_LP'):
        '''
        fit the large-scale parameters which describes errors 
        '''
        #
        Surf_coeff=T.tensor(Init).to(Device)
        test=correctphase2(Meas_maps,DEVICE=Device)
        Log={'residual':[]}
        def lossfuc(parameters):
            '''Change the input parameters into 'tensor type' '''
            Params=T.tensor(parameters,requires_grad=True)
            paraA=Params[0:6*Map_N].to(Device)
            paraP=Params[6*Map_N:].to(Device)

            # Beam Calculation
            Data=self.FF(Surf_coeff,paraA,paraP)
            Data=correctphase2(Data,DEVICE=Device)
            # residual
            r=((Data-test)**2).sum()
            
            print('res:',r.item())
            Log['residual'].append(r.item())
            r.backward()
            return r.data.cpu().numpy(),Params.grad.data.cpu().numpy()       
        fit_coeff=np.append(np.array([1.0,0,0,0,0,0]*Map_N),np.array([0,0,0,0,0]*Map_N))
        start=time.perf_counter()
        self.result_LP=scipy.optimize.minimize(lossfuc,fit_coeff,method='BFGS',jac=True,tol=1e-4)
        elapsed=(time.perf_counter()-start)
        print('Cost time:', elapsed)
        with h5py.File(self.output_folder+'/'+outputfilename+'.h5py','w') as f:
            for item in dir(self.result_LP):
                f.create_dataset(item,data=self.result_LP[item])
            f.create_dataset('log',data=np.array(Log['residual']))
            f.create_dataset('time',data=elapsed)

    def fit_surface(self,Meas_maps,constraint=[1,1,1,1,1,1],
                    Device=T.device('cpu'),
                    Init_LP=np.append(np.array([1,0,0,0,0,0]*5),np.array([0,0,0,0,0]*5)),
                    Map_N=5,
                    outputfilename='fit_adjusters'):
        ''' fit the surface profiles of the two mirrors'''
        ''' Init_LP is initial input parameters for the large-scale errors in aperture plane.
            Here we suggest to use the values from the first fit_LP fitting.
        '''
        Lambda_00=constraint[0]
        Lambda_10=constraint[1]
        Lambda_01=constraint[2]
        Lambda_20=constraint[3]
        Lambda_02=constraint[4]
        Lambda0=constraint[5]
        x2,y2,x1,y1=DATA2TORCH(self.S2_x,self.S2_y,self.S1_x,self.S1_y,DEVICE=Device)
        test=correctphase2(Meas_maps,DEVICE=Device)
        Log={'residual':[]}
        def lossfuc(parameters):
            '''Change the input parameters into 'tensor type' '''
            Params=T.tensor(parameters,requires_grad=True)
            parameters=Params.to(Device)
            Surf_coeff=parameters[0:-(6+5)*Map_N]
            paraA=parameters[-(6+5)*Map_N:-5*Map_N]
            paraP=parameters[-5*Map_N:]

            Data=self.FF(Surf_coeff,paraA,paraP)
            Data=correctphase2(Data,DEVICE=Device)

            r0=((Data-test)**2).sum()
            # consider the lagrange factors
            S2=Surf_coeff[0:5*69]
            S1=Surf_coeff[5*69:]   
            Z_00=T.abs((S1).sum())+T.abs((S2).sum()) # compress piston error in large scale;
            Z_10=T.abs((x2*S2).sum())+T.abs((x1*S1).sum()) # compress slope error in x
            Z_01=T.abs((y2*S2).sum())+T.abs((y1*S1).sum())# slope error in y
            Z_20=T.abs((S2*x2**2).sum())+T.abs((S1*(x1**2)).sum()) #  curvature error;
            Z_02=T.abs((S2*y2**2).sum())+T.abs((S1*(y1**2)).sum()) 
            Z=(S2**2).mean()+(S1**2).mean()
            r=r0+Lambda_00*Z_00+Lambda_10*Z_10+Lambda_01*Z_01+Lambda_20*Z_20+Lambda_02*Z_02+Lambda0*Z
            print(Z_00.item(),Z_10.item(),Z_01.item(),Z_20.item(),Z_02.item())
            print(Z.item(),r0.item())
            r=r.sum()
            r.backward()
            print('res:',r.item())
            return r.data.cpu().numpy(),Params.grad.data.cpu().numpy()
        
        fit_coeff=np.append(np.zeros(5*(69+77)),Init_LP).ravel()

        start=time.perf_counter()
        self.result=scipy.optimize.minimize(lossfuc,fit_coeff,method="BFGS",jac=True,tol=1e-4)
        elapsed=(time.perf_counter()-start)
        print('Cost time:', elapsed)

        with h5py.File(self.output_folder+'/'+outputfilename+'.h5py','w') as f:
            for item in dir(self.result):
                f.create_dataset(item,data=self.result[item])
            f.create_dataset('residual',data=np.array(Log['residual']))
            f.create_dataset('time',data=elapsed)


    def fit_surface_zk(self,Meas_maps,constraint=[1,1],
                    Device=T.device('cpu'),
                    Init_LP=np.append(np.array([1,0,0,0,0,0]*5),np.array([0,0,0,0,0]*5)),
                    Map_N=5,
                    outputfilename='fit_adjusters_zk'):
        ''' fit the surface profiles of the two mirrors'''
        ''' Init_LP is initial input parameters for the large-scale errors in aperture plane.
            Here we suggest to use the values from the first fit_LP fitting.
        '''
        Lambda0=constraint[0]
        Lambda1=constraint[1]
        #x2,y2,x1,y1=DATA2TORCH(self.S2_x,self.S2_y,self.S1_x,self.S1_y,DEVICE=Device)
        test=correctphase2(Meas_maps,DEVICE=Device)
        Log={'residual':[]}
        def lossfuc(parameters):
            '''Change the input parameters into 'tensor type' '''
            Params=T.tensor(parameters,requires_grad=True)
            parameters=Params.to(Device)
            params=parameters[0:-(6+5)*Map_N].reshape(2,-1)
            Surf_coeff=T.cat((T.zeros((2,3)),params),axis=1)
            paraA=parameters[-(6+5)*Map_N:-5*Map_N]
            paraP=parameters[-5*Map_N:]

            Data=self.FF(Surf_coeff,paraA,paraP)
            Data=correctphase2(Data,DEVICE=Device)

            r0=((Data-test)**2).sum()
            # consider the lagrange factors

            Z2=(T.abs(Surf_coeff[0,:])**2).sum()
            Z1=(T.abs(Surf_coeff[1,:])**2).sum()
            r=r0+Lambda0*Z2+Lambda1*Z1
            print(Z2.item(),Z1.item(),r0.item())
            r=r.sum()
            r.backward()
            print('res:',r.item())
            return r.data.cpu().numpy(),Params.grad.data.cpu().numpy()
        
        fit_coeff=np.append(np.zeros(2*(poly_N(self.Z_order)-3)),Init_LP).ravel()

        start=time.perf_counter()
        self.result_zk=scipy.optimize.minimize(lossfuc,fit_coeff,method="BFGS",jac=True,tol=1e-4)
        elapsed=(time.perf_counter()-start)
        print('Cost time:', elapsed)

        with h5py.File(self.output_folder+'/'+outputfilename+'.h5py','w') as f:
            for item in dir(self.result_zk):
                f.create_dataset(item,data=self.result_zk[item])
            f.create_dataset('residual',data=np.array(Log['residual']))
            f.create_dataset('time',data=elapsed)

    