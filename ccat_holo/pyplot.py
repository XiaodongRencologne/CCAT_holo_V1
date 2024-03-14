# %%
import matplotlib.pyplot as plt
import numpy as np
def plot_beamcontour(x,y,
                     Meas_beams,
                     ref=None,
                     Map_size=0.15,
                     levels=[-30,-20,-15,-10,-5,-2,0],
                     outputfilename='beam_comparison_contourPlot.png'):
    fig,ax=plt.subplots(figsize=(18.5,3.5),ncols=5,nrows=1)
    fig.tight_layout(pad=3)
    labelsize=9
    # individual beams
    N=5
    N_size=int(np.sqrt(Meas_beams[0,:].size))
    p0=np.sqrt((np.abs(Meas_beams[0,:]+1j*Meas_beams[1,:])**2).sum())
    beams=Meas_beams/p0
    # ref beams

    p0=np.sqrt((np.abs(ref[0,:]+1j*ref[1,:])**2).sum())
    Ref=ref/p0
    
    # plot setup
    for n in range(5):
        ax[n].set_xlim([x.min(),x.max()])
        ax[n].set_ylim([y.min(),y.max()])
        ax[n].set_xticks(np.linspace(x.min(),x.max(),5))
        ax[n].set_yticks(np.linspace(x.min(),x.max(),5))
        ax[n].tick_params(direction='in',labelsize=labelsize)
    
    '''
    # draw reference beams
    p0=ax[2].pcolor(x,y,
                    np.log10(np.abs((Ref[0,:]+1j*Ref[1,:]).reshape(N_size,-1)))*20,
                    cmap='jet')
    p1=ax[0].pcolor(x,y,np.log10(np.abs((Ref[2,:]+1j*Ref[3,:]).reshape(N_size,-1)))*20,
                    cmap='jet')
    p2=ax[1].pcolor(x,y,np.log10(np.abs((Ref[4,:]+1j*Ref[5,:]).reshape(N_size,-1)))*20,
                    cmap='jet')
    p3=ax[3].pcolor(x,y,np.log10(np.abs((Ref[6,:]+1j*Ref[7,:]).reshape(N_size,-1)))*20,
                    cmap='jet')
    p4=ax[4].pcolor(x,y,np.log10(np.abs((Ref[8,:]+1j*Ref[9,:]).reshape(N_size,-1)))*20,
                    cmap='jet')
    '''
    # plot the center beams
    p0=ax[2].contour(x,y,np.log10(np.abs((beams[0,:]+1j*beams[1,:]).reshape(N_size,-1)))*20,
                     levels,linewidths=(2,),colors=('r',))
    p1=ax[0].contour(x,y,np.log10(np.abs((beams[2,:]+1j*beams[3,:]).reshape(N_size,-1)))*20,
                     levels,linewidths=(2,),colors=('r',))
    p2=ax[1].contour(x,y,np.log10(np.abs((beams[4,:]+1j*beams[5,:]).reshape(N_size,-1)))*20,
                     levels,linewidths=(2,),colors=('r',))
    p3=ax[3].contour(x,y,np.log10(np.abs((beams[6,:]+1j*beams[7,:]).reshape(N_size,-1)))*20,
                     levels,linewidths=(2,),colors=('r',))
    p4=ax[4].contour(x,y,np.log10(np.abs((beams[8,:]+1j*beams[9,:]).reshape(N_size,-1)))*20,
                     levels,linewidths=(2,),colors=('r',))
    
    
    #ax[0].set_title('Simulated ideal focused beam',color='k',fontsize=12)
    
    p0=ax[2].contour(x,y,np.log10(np.abs((Ref[0,:]+1j*Ref[1,:]).reshape(N_size,-1)))*20,
                    levels,linewidths=(2,),colors=('b',))
    p1=ax[0].contour(x,y,np.log10(np.abs((Ref[2,:]+1j*Ref[3,:]).reshape(N_size,-1)))*20,
                    levels,linewidths=(2,),colors=('b',))
    p2=ax[1].contour(x,y,np.log10(np.abs((Ref[4,:]+1j*Ref[5,:]).reshape(N_size,-1)))*20,
                    levels,linewidths=(2,),colors=('b',))
    p3=ax[3].contour(x,y,np.log10(np.abs((Ref[6,:]+1j*Ref[7,:]).reshape(N_size,-1)))*20,
                    levels,linewidths=(2,),colors=('b',))
    p4=ax[4].contour(x,y,np.log10(np.abs((Ref[8,:]+1j*Ref[9,:]).reshape(N_size,-1)))*20,
                    levels,linewidths=(2,),colors=('b',))
    
    plt.savefig(outputfilename)
    plt.show()
