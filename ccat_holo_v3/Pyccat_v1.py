import numpy as np
import torch as T
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

from Kirchhoffpy.inference import DATA2TORCH, correctphase2
from Kirchhoffpy.inference import fitting_func, fitting_func_zernike


# surface errors expressed by zernike polynomials.
c=299792458*1000

holo_setup={'Rx0':([0,0,600],'beam/on-axis.txt'),
            'Rx1':([400,400,600],'beam/400_400_600.txt'),
            'Rx2':([400,-400,600],'beam/400_-400_600.txt'),
            'Rx3':([-400,400,600],'beam/-400_400_600.txt'),
            'Rx4':([-400,-400,600],'beam/-400_-400_600.txt')
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
        self.output_folder=output_folder
        self.holo_conf=holo_conf
        self.output_filename=None
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
        parameters=np.genfromtxt(Model_folder+'/input.txt',delimiter=',')[:,1]

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
        # beam profile of the feedhorn of the receiver
        self.input_feed_beam=input_Rx_beam
        
        self._coords(Rx=[0,0,0])

    def _coords(self,Rx=[0,0,0]):
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

    def _beam(self,scan_file,Rx=[0,0,0],Matrix=False,S2_init=np.zeros((5,69)),S1_init=np.zeros((5,77)),Error_m2=0,Error_m1=0):
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
        Matrix1,Field_fimag,cosm2_r=PO_scalar(m2,m2_n,
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
                                        Field_fimag,
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
        Matrix3,Field_s,cosm1_r=PO_scalar(m1,m1_n,self.m1_dA,
                                          source,cosm1_i,
                                          Field_m1,
                                          self.k,
                                          Keepmatrix=Matrix
                                          )
        
        elapsed =(time.perf_counter()-start)
        print('time used:',elapsed)
        # Save the computation data into h5py file, and the intermediate Matrixs that wil
        # be used for accelerating the forward beam calculations.
        self.output_filename=self.output_folder+'/data_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
        with h5py.File(self.output_filename,'w') as f:
              #f.create_dataset('conf',data=(Rx,scan_file))
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
              f.create_dataset('F_if_real',data=Field_fimag.real)
              f.create_dataset('F_if_imag',data=Field_fimag.imag)
              f.create_dataset('F_beam_real',data=Field_s.real)
              f.create_dataset('F_beam_imag',data=Field_s.imag)
              f.create_dataset('aperture',data=self.aperture_xy)
              f.create_dataset('scan_pattern',data=np.concatenate((source.x,
                                                                   source.y,source.z)).reshape(3,-1))

    def plot_beam(self,filename=None):
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
            print(beam.real)
            print(beam.imag)

    def build_holo(self,S2_init=np.zeros((5,69)),S1_init=np.zeros((5,77))):
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
                self._beam(self.holo_conf[keys][1],Rx=self.holo_conf[keys][0],Matrix=True,S2_init=S2_init,S1_init=S1_init)
    
    def mk_FF_5maps(self,fitting_param='panel adjusters',Device=T.device('cpu'),Z_order=7):
        '''Define forward function (FF) !
           *** fitting_param ***  1. 'panel adjusters' the fitting parameters are adjuster movements.
                                   2. 'zernike' fit the coefficients of a set of zernike polynomials
                                      for large spatial scale surface errors.
                                      'zernike' model is chosen, maximum zernike order 'Z_order' 
                                      must be given.
        '''
        # read the aperture field points coordinates
        Rx=self.holo_conf['Rx0'][0]
        file=self.output_folder+'/data_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
        with h5py.File(file,'r') as f:
            self.aperture_xy=f['aperture'][:]
        Aperture=DATA2TORCH(self.aperture_xy,DEVICE=Device)
        List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1=DATA2TORCH(self.Panel_center_M2,self.Panel_center_M1,
                                                             self.m2_0,self.m1_0,
                                                             self.p_m2,self.q_m2,
                                                             self.p_m1,self.q_m1,DEVICE=Device)
        Vars={'Rx0': None, 'Rx1': None, 'Rx2': None, 'Rx3': None, 'Rx4': None}
        for i in range(5):
            Rx=self.holo_conf['Rx'+str(i)][0]
            file=self.output_folder+'/data_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
            Vars['Rx'+str(i)]=DATA2TORCH(*Load_Mat(file))
        
        if fitting_param=='panel adjusters':
            def FF(Adjusters,Para_A,Para_p):
                R0=fitting_func(*Vars['Rx0'],Adjusters,List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,Para_A[0:6],Para_p[0:5],Aperture)
                R1=fitting_func(*Vars['Rx1'],Adjusters,List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,Para_A[6:6*2],Para_p[5:5*2],Aperture)
                R2=fitting_func(*Vars['Rx2'],Adjusters,List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,Para_A[6*2:6*3],Para_p[5*2:5*3],Aperture)
                R3=fitting_func(*Vars['Rx3'],Adjusters,List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,Para_A[6*3:6*4],Para_p[5*3:5*4],Aperture)
                R4=fitting_func(*Vars['Rx4'],Adjusters,List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,Para_A[6*4:],Para_p[5*4:],Aperture)
                return T.cat((R0,R1,R2,R3,R4))
            self.FF=FF
        elif fitting_param=='zernike':
            self.Z_surf2=make_zernike(Z_order,m2.x/self.R2,(m2.y+0.0)/self.R2,dtype='torch',device=DEVICE)
            self.Z_surf1=make_zernike(Z_order,m1.x/self.R1,(m1.y-0.0)/self.R1,dtype='torch',device=DEVICE)
            def FF(Z_coeff2,Z_coeff1,Para_A,Para_p):
                R0=fitting_func_zernike(*Vars['Rx0'],Z_coeff2,Z_coeff1,List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,Para_A[0:6],Para_p[0:5],Aperture)
                R1=fitting_func_zernike(*Vars['Rx1'],Z_coeff2,Z_coeff1,List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,Para_A[6:6*2],Para_p[5:5*2],Aperture)
                R2=fitting_func_zernike(*Vars['Rx2'],Z_coeff2,Z_coeff1,List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,Para_A[6*2:6*3],Para_p[5*2:5*3],Aperture)
                R3=fitting_func_zernike(*Vars['Rx3'],Z_coeff2,Z_coeff1,List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,Para_A[6*3:6*4],Para_p[5*3:5*4],Aperture)
                R4=fitting_func_zernike(*Vars['Rx4'],Z_coeff2,Z_coeff1,List_2,List_1,m2,m1,p_m2,q_m2,p_m1,q_m1,self.k,Para_A[6*4:],Para_p[5*4:],Aperture)
                return T.cat((R0,R1,R2,R3,R4))
            self.FF=FF
        else:
            print('The forward function is not sucessfully created!!!!')
        




            

        
        
        





"""    
    def mk_FF(self,conf=['Rx0','Rx1','Rx2','Rx3','Rx4'],parameter_type='panel adjusters',Z_order=7):
        '''Define forward function (FF) !
           *** conf *** Choose the beams that will be used for holographic analysis!
           *** parameter_type ***  1. 'panel adjusters' the fitting parameters are adjuster movements.
                                   2. 'zernike' fit the coefficients of a set of zernike polynomials
                                      for large spatial scale surface errors.
                                      'zernike' model is chosen, maximum zernike order 'Z_order' 
                                      must be given.
        '''
        # check if all given receiver positions are pre-defined and -calculated. 
        status=True
        for item in conf:
            if item in self.holo_conf.keys():
                pass
            else:
                status=False
                print('The selected Rx position is not defined!!!')
        if not status:
            print('!!! forward function is not sucessfully built!!!!!')
            pass
        else:
            #read the pre-calculated matrix
            # read the aperture x y position.
            Rx=self.holo_conf[conf[0]][0]
            file1=self.output_folder+'/data_Rx_dx'+str(Rx[0])+'_dy'+str(Rx[1])+'_dz'+str(Rx[2])+'.h5py'
            with h5py.File(file1,'r') as f:
                self.aperture_xy=f['aperture'][:]
            
            List_2,List_1,m2,m1,aperture,p_m2,q_m2,p_m1,q_m1=DATA2CUDA(List_m2,List_m1,m2,m1,aperture,p_m2,q_m2,p_m1,q_m1,DEVICE=DEVICE)


            if parameter_type=='panel adjusters':
                pass
            elif parameter_type=='zernike':
                pass
    
    
    def Forward_Fnc(self,S_m2,S_m1,Apert_amp,Apert_phase,):
        pass
"""