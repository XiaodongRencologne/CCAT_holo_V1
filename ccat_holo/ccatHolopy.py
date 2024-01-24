import numpy as np
import torch as T

from Pyccat import Make_fitfuc_1,Make_fitfuc_4,Make_fitfuc_5
from Pyccat import Make_fitfuc_zernike1,Make_fitfuc_zernike4,Make_fitfuc_zernike5

class CCATholo():
    def __init__(self,Model_folder,scan_folder):
        self.forward=None
        self.inputfolder=Model_folder
        self.ad2_x=None
        self.ad2_y=None
        self.ad1_x=None
        self.ad1_y=None

    def Make_fitfuc(self,scan_folder,
                    defocus,
                    Rx_p=[[0,0]],
                    init_ad=np.zeros(5*(69+77)),
                    fitting_type='zernike',
                    zernike_order=7,
                    DEVICE=T.device('cpu')):
        Num=len(Rx_p)
        if Num==1:
            self.forward,self.ad2_x,self.ad2_y,self.ad1_x,self.ad1_y=Make_fitfuc_grid_oneBeam(self.inputfolder,
                                                 scan_folder,
                                                 [Rx_p[0][0],Rx_p[0][1],defocus],
                                                 init_ad[0:5*69],
                                                 init_ad[5*69:],
                                                 DEVICE=DEVICE)
        elif Num==4:
            if fitting_type.lower=='zernike':
                self.forward=Make_fitfuc_zernike(self.inputfolder,
                                                scan_folder,
                                                [Rx_p[0][0],Rx_p[0][1],defocus],
                                                [Rx_p[1][0],Rx_p[1][1],defocus],
                                                [Rx_p[2][0],Rx_p[2][1],defocus],
                                                [Rx_p[3][0],Rx_p[3][1],defocus],
                                                init_ad[0:5*69],
                                                init_ad[5*69:],
                                                zernike_order,
                                                DEVICE=DEVICE)
            elif fitting_type.lower=='adjuster':
                self.forward,self.ad2_x,self.ad2_y,self.ad1_x,self.ad1_y=Make_fitfuc(self.inputfolder,
                                        scan_folder,
                                        [Rx_p[0][0],Rx_p[0][1],defocus],
                                        [Rx_p[1][0],Rx_p[1][1],defocus],
                                        [Rx_p[2][0],Rx_p[2][1],defocus],
                                        [Rx_p[3][0],Rx_p[3][1],defocus],
                                        init_ad[0:5*69],
                                        init_ad[5*69:],
                                        DEVICE=DEVICE)
        elif Num==5:
            if fitting_type.lower=='zernike':
                self.forward=Make_fitfuc_zernike5(self.inputfolder,
                                                scan_folder,
                                                [Rx_p[0][0],Rx_p[0][1],defocus],
                                                [Rx_p[1][0],Rx_p[1][1],defocus],
                                                [Rx_p[2][0],Rx_p[2][1],defocus],
                                                [Rx_p[3][0],Rx_p[3][1],defocus],
                                                [Rx_p[4][0],Rx_p[4][1],defocus],
                                                init_ad[0:5*69],
                                                init_ad[5*69:],
                                                zernike_order,
                                                DEVICE=DEVICE)
            elif fitting_type.lower=='adjuster':
                self.forward,self.ad2_x,self.ad2_y,self.ad1_x,self.ad1_y=Make_fitfuc_5(self.inputfolder,
                                        scan_folder,
                                        [Rx_p[0][0],Rx_p[0][1],defocus],
                                        [Rx_p[1][0],Rx_p[1][1],defocus],
                                        [Rx_p[2][0],Rx_p[2][1],defocus],
                                        [Rx_p[3][0],Rx_p[3][1],defocus],
                                        [Rx_p[4][0],Rx_p[4][1],defocus],
                                        init_ad[0:5*69],
                                        init_ad[5*69:],
                                        DEVICE=DEVICE)
    def read_Meas_Data(self,M_data_files):
        '''M_data_files={'filename1','filename2',
        'filename3','filename4','filename5'}'''
        Num=len(M_data_files)
        if Num==1:
            center=np.genfromtxt(M_data_files[0],delimiter=',').T
            self.testdata=center
        elif Num==4:
            pospos=np.genfromtxt(M_data_files[0],delimiter=',').T
            posneg=np.genfromtxt(M_data_files[1],delimiter=',').T
            negpos=np.genfromtxt(M_data_files[2],delimiter=',').T
            negneg=np.genfromtxt(M_data_files[3],delimiter=',').T
            self.testdata=np.concatenate((pospos,posneg,negpos,negneg)).reshape(8,-1)
        elif Num==5:
            pospos=np.genfromtxt(M_data_files[0],delimiter=',').T
            posneg=np.genfromtxt(M_data_files[1],delimiter=',').T
            negpos=np.genfromtxt(M_data_files[2],delimiter=',').T
            negneg=np.genfromtxt(M_data_files[3],delimiter=',').T
            center=np.genfromtxt(M_data_files[3],delimiter=',').T
            self.testdata=np.concatenate((pospos,posneg,negpos,negneg,center)).reshape(10,-1)

    def fit_Sys_Error_Onebeam(self,DEVICE=T.device('cpu')):
        '''
        '''
        testdata=T.tensor(self.testdata)
        testdata=correctphase2(testdata)
        testdata=testdata.to(DEVICE)
        Adjustors=T.tensor(np.zeros(5*(69+77))).to(DEVICE)
        def loss_fuc(fitting_params):
            '''input parameters put to tensor type'''
            Params=T.tensor(parameters,requires_grad=True).to(DEVICE)
            '''large scale error in amplitude'''
            paraA=Params[0:6]
            '''large scale error in phase term (pointing error, curvature erorr)'''
            paraP=Params[6:]
            '''forward calculation'''        
            Data=self.forward(Adjustors,paraA,paraP)
            Data=correctphase2(Data)
            '''residual between simulation and measurement'''
            r=((Data-self.testdata)**2).sum()
            r.backward()
            print(r.item())
            return r.data.cpu().numpy(),Params.grad.data.cpu().numpy()
        ad=np.array([1.0,0,0,0,0,0,0,0,0,0,0])
        results=scipy.optimize.minimize(loss_fuc,ad,method='BFGS',jac=True,tol=1e-5)
        return results
    
    def fit_Mirrors_Onebeam(self,Large_param,DEVICE=T.device('cpu')):
        x2=T.tensor(self.ad2_x).to(DEVICE)
        y2=T.tensor(self.ad2_y).to(DEVICE)
        x1=T.tensor(self.ad1_x).to(DEVICE)
        y1=T.tensor(self.ad1_y).to(DEVICE)
        def loss_fuc(fitting_params):
            '''input parameters put to tensor type'''
            Params=T.tensor(fitting_params,requires_grad=True).to(DEVICE)
            '''CPU OR GPU'''
            '''adjusters'''
            '''large scale error in amplitude'''
            #paraA=T.tensor(np.array([1,0,0,0,0,0]*4)).to(DEVICE0)
            paraA=Params[0:6]
            paraP=Params[6:6+5]
            '''large scale error in phase term (pointing error, curvature erorr)'''
            #paraP=T.tensor(np.array([0,0,0,0,0]*4)).to(DEVICE0)
            '''forward calculation'''
            parameters=T.cat((T.zeros(5*69,dtype=T.float64),Params[6+5:]))
            Data=self.forward(parameters,paraA,paraP,DEVICE=DEVICE)
            Data=correctphase2(Data)
            '''residual between simulation and measurement'''
            r0=((Data-self.testdata)**2).sum()
            
            S2=parameters[:5*69]
            S1=parameters[5*69:]
            #print(S2.shape,S1.shape)
            Z_00=T.abs((S1).sum())+T.abs((S2).sum()); # compress piston error in large scale;
            Z_10=T.abs((x2*S2).sum())+T.abs((x1*S1).sum()) # compress slope error in x
            Z_01=T.abs((y2*S2).sum())+T.abs((y1*S1).sum());# slope error in y
            Z_20=T.abs((S2*x2**2).sum())+T.abs((S1*(x1**2)).sum()); #  curvature error;
            Z_02=T.abs((S2*y2**2).sum())+T.abs((S1*(y1**2)).sum()); 
            Z=(S2**2).mean()+(S1**2).mean()
            #r=r0+Lambda_00*Z_00+Lambda_10*Z_10+Lambda_01*Z_01+Lambda_20*Z_20+Lambda_02*Z_02;

            
            r=r0+Z
            r=r.sum()
            print(r.item());
            print(Z_00.item(),Z_10.item(),Z_01.item(),Z_20.item(),Z_02.item())
            r.backward()

            return r.data.cpu().numpy(),Params.grad.data.cpu().numpy()
        ad=np.append(Large_param,np.zeros(5*(69+77)))
        start=time.perf_counter();
        results=scipy.optimize.minimize(fitfuc,ad,method="BFGS",jac=True,tol=1e-6)
        elapsed =(time.perf_counter()-start)
        print('time used:',elapsed)
        return results

