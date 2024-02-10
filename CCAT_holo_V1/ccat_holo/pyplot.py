# %%
import matplotlib.pyplot as plt
# %%
def plot_beammaps(Meas_beams,compared_to=None):
    

def plot_beamcontour(Meas_beams,
                     compared_to=None,
                     Map_size=0.15,
                     levels=[-30,-20,-15,-10,-5,-2,0]):
    fig,ax=plt.subplots(figsize=(18.5,3.5),ncols=5,nrows=1)
    fig.tight_layout(pad=3)
    labelsize=9
    N=int(Meas_beams.shape[0]/2)
    N_size=int(np.sqrt(Meas_beams.shape[1]))
    if compared_to!=None:
        D0=compared_to
        for n in range(N):
            d=
            p0=ax[n].contour(x0,y0,np.log10(np.abs(D0))*20,levels,linewidths=(2,),colors=('b',));
            p0=ax[n].contour(x0,y0,np.log10(np.abs(d1))*20,levels,linewidths=(1,),colors=('r',));
            #ax[0].set_title('Simulated ideal focused beam',color='k',fontsize=12)
            ax[n].set_xlim([-L_range/2,L_range/2])
            ax[n].set_ylim([-L_range/2,L_range/2])
            ax[n].set_xticks(np.linspace(-L_range/2,L_range/2,5))
            ax[n].set_yticks(np.linspace(-L_range/2,L_range/2,5))
            ax[n].tick_params(direction='in',labelsize=labelsize)
    
    for n in range(N):
        pass

