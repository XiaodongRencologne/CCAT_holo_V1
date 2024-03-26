# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch as T
from ccat_holo.Pyccat import CCAT_holo

Ref_status=False
DEVICE0=T.device('cpu')
#%%
Rx=[0,0,600]
Beam_points='CCAT_model/beam/on-axis.txt'
'''Define the CCAT_holo model'''
# beamfolder
Folder='SamplingN'
Model=CCAT_holo('CCAT_model',Folder)
Model.M2_N=[60,60]
Model.M1_N=[60,100]
Model.fimag_size=[700,700]
Model.fimag_N=[201,201]
# %%
''' create a reference beam data for Rx at [0,0,600] (out-of-focus case)'''
if Ref_status:
    Model._beam(Beam_points,Rx=Rx,file_name='Ref_N_100')
    #Model.plot_beam()
    S_ref=Model.Field_s.real+1j*Model.Field_s.imag
    IF_ref=Model.Field_fimag.real+1j*Model.Field_fimag.imag
else:
    Data=h5py.File('SamplingN/Ref_N_100_Rx_dx0_dy0_dz600.h5py','r')
    S_ref=Data['F_beam_real'][:]+1j*Data['F_beam_imag'][:]
    IF_ref=Data['F_if_real'][:]+1j*Data['F_if_imag'][:]
# %%
"""
'''start the sampling points test!!'''
N_m2=list(range(10,21))
N_m1=list(range(10,21))
with h5py.File('SamplingN/residual.h5py','w-') as f:
    for n in N_m2:
        Model.M2_N=[n,n]
        for m in N_m1:
            print(n,m)
            Model.M1_N=[m,m]
            Model._beam(Beam_points,Rx=Rx,file_name='M2_'+str(n)+'M1_'+str(m))
            re=Model.Field_s.real-S_ref.real
            im=Model.Field_s.imag-S_ref.imag
            f.create_dataset('M2_'+str(n)+'M1_'+str(m)+'s',data=re+1j*im)
            re=Model.Field_fimag.real-IF_ref.real
            im=Model.Field_fimag.imag-IF_ref.imag
            f.create_dataset('M2_'+str(n)+'M1_'+str(m)+'if',data=re+1j*im)
"""
# %%
N_m2=list(range(10,21))
N_m1=list(range(10,21))
with h5py.File('SamplingN/residual.h5py','w') as f:
    r_s=np.array([])
    r_if=np.array([])
    for n in N_m2:
        Model.M2_N=[n,n]
        for m in N_m1:
            Model.M1_N=[m,m]
            data=h5py.File(Folder+'/M2_'+str(n)+'M1_'+str(m)+'_Rx_dx0_dy0_dz600.h5py')
            re=data['F_beam_real'][:]-S_ref.real
            im=data['F_beam_imag'][:]-S_ref.imag
            f.create_dataset('M2_'+str(n)+'M1_'+str(m)+'s',data=re+1j*im)
            r_s=np.append(r_s,(re**2+im**2).mean()/(np.abs(S_ref)**2).sum())

            re=data['F_if_real'][:]-IF_ref.real
            im=data['F_if_imag'][:]-IF_ref.imag
            f.create_dataset('M2_'+str(n)+'M1_'+str(m)+'if',data=re+1j*im)
            r_if=np.append(r_if,(re**2+im**2).mean()/(np.abs(IF_ref)**2).max())
            data.close()
    f.create_dataset('r_s',data=r_s)
    f.create_dataset('r_if',data=r_if)

# %%
data=h5py.File('SamplingN/residual.h5py','r')
rs=data['r_s'][:].reshape(11,11)
rIF=data['r_if'][:].reshape(11,11)
data.close()
# %%
fig=plt.figure(figsize=(6,5))
plt.contourf(N_m2,N_m1,np.log10(rs)*10)
plt.axis('equal')
plt.colorbar()
# %%
fig=plt.figure(figsize=(5,5))
plt.plot(10*np.log10(rs[:,:]),'*-')
# %%
fig=plt.figure(figsize=(6,5))
plt.contourf(N_m2,N_m1,np.log10(rIF)*10)
plt.axis('equal')
plt.colorbar()
# %%
fig=plt.figure(figsize=(5,5))
plt.plot(10*np.log10(rIF[:,:]),'*-')

# %%
