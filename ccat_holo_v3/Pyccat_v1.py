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
    def __init__(self,Model_folder):
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

        M2_size=parameters[0:2]
        M1_size=parameters[2:4]
        M2_N=parameters[10:12]
        M1_N=parameters[12:14]

        self.m2,self.m2_n,self.m2_dA=squarepanel(self.Panel_center_M2[...,0],self.Panel_center_M2[...,1],
                                                    M2_size[0],M2_size[1],
                                                    M2_N[0],M2_N[1],
                                                    self.surface_m2
                                                    )
        self.m1,self.m1_n,self.m1_dA=squarepanel(self.Panel_center_M1[...,0],self.Panel_center_M2[...,1],
                                                    M1_size[0],M1_size[1],
                                                    M1_N[0],M1_N[1],
                                                    surface_m1
                                                    )
        # panel adjuster position
        self.p_m2=parameters[6]
        self.q_m2=parameters[7]
        self.p_m1=parameters[8]
        self.q_m1=parameters[9]

        self.S_m2_xy=None
        self.S_m1_xy=None
        
        # Intermedia plane
        fimag_N=parameters[14:16]
        fimag_size=parameters[16:18]
        self.fimag,self.fimag_n,self.fimag_dA=ImagPlane(fimag_size[0],fimag_size[1],
                                                        fimag_N[0],fimag_N[1]  
                                                        )
        '''Electrical parameters'''
        electro_params=np.genfromtxt(Model_folder+'/electrical_parameters.txt',delimiter=',')[...,1]
        self.freq=electro_params[0]*10**9
        self.edge_taper=electro_params[1]
        self.Angle_taper=electro_params[2]/180*np.pi
        self.Lambda=c/self.freq
        self.k=2*np.pi/self.Lambda

        
        self._coords(defocus=[0,0,0])

    def coords(self,defocus=[0,0,0]):
        '''coordinates systems'''
        '''
        #angle# is angle change of local coordinates and global coordinates;
        #D#     is the distance between origin of local coord and global coord in global coordinates;
        '''
        '''
        some germetrical parametrs
        '''
        self.Theta_0  =  0.927295218001612; # offset angle of MR;
        self.Ls       =  12000.0;           # distance between focal point and SR
        self.Lm       =  6000.0;            # distance between MR and SR;
        self.L_fimag  = 18000+self.Ls
        self.F        = 20000               # equivalent focal length of M2

        self.angle_m2=[-(np.pi/2+self.Theta_0)/2,0,0] #  1. m2 and global co-ordinates
        self.D_m2=[0,-Lm*np.sin(self.Theta_0),0]
        
        self.angle_m1=[-self.Theta_0/2,0,0]          #  2. m1 and global co-ordinates
        self.D_m1=[0,0,self.Lm*np.cos(self.Theta_0)]
        
        self.angle_s=[0,np.pi,0];               #  3. source and global co-ordinates
        self.D_s=[0,0,0]
        
        self.angle_fimag=[-self.Theta_0,0,0];        #  4. fimag and global co-ordinates
        self.defocus_fimag=[0,0,0]
        self.defocus_fimag[2]=1/(1/self.F-1/(self.Ls+self.defocus[2]))+self.L_fimag
        self.defocus_fimag[1]=(F+L_fimag-defocus_fimag[2])/F*defocus[1]
        self.defocus_fimag[0]=(F+L_fimag-defocus_fimag[2])/F*defocus[0]
        self.D_fimag=[0,0,0]
        self.D_fimag[0]=self.defocus_fimag[0]
        self.D_fimag[1]=self.defocus_fimag[1]*np.cos(self.Theta_0)\
            -np.sin(self.Theta_0)*(self.L_fimag-self.defocus_fimag[2]+self.Lm)
        self.D_fimag[2]=-self.defocus_fimag[1]*np.sin(self.Theta_0)\
            -np.cos(self.Theta_0)*(self.L_fimag-self.defocus_fimag[2])
        
        # feed coordinate system
        '''
        C=1/(1/Lm-1/F)+defocus[2]+Ls;
        C=21000;
        angle_f=[np.pi/2-defocus[1]/C,0,-defocus[0]/C]; 
        D_f=[defocus[0],Ls+defocus[2]-Lm*np.sin(Theta_0),-defocus[1]];
        '''
        self.angle_f=[np.pi/2,0,0];    
        self.D_f=[defocus[0],Ls+defocus[2]-self.Lm*np.sin(self.Theta_0),-defocus[1]]

    def _build_CCAT(self):
        pass



















         