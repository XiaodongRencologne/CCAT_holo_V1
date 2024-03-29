# %%
import sys,os
sys.path.append('..')
from ccat_holo.BeamPattern import squarePattern
# %%
u0=0.027
v0=0.027
urange=0.01
vrange=0.01
Nu=501
Nv=501
forder='501/'
if not os.path.exists(forder):
    os.makedirs(forder)
print(os.path.exists(forder))

squarePattern(u0,v0,urange,vrange,Nu,Nv,file=forder,distance=300000,Type='on-axis')
squarePattern(u0,v0,urange,vrange,Nu,Nv,file=forder,distance=300000,Type='off-axis')


# %%
