# #### Constant setpoint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'

loc = 'ConstantSetpoint.xlsx'
index = [2,4,7,11,16]
time = [2,4,6,10,14]

data_state = np.zeros((6,50,50))
data_input = np.zeros((6,50,50))

fig = plt.figure(32,figsize=(24,8))
kk = 0
for i in index:
    data_state = pd.read_excel(loc, header=None, sheet_name='State' + str(i)).values # can scale it as: * 10 + 20
    data_input = pd.read_excel(loc, header=None, sheet_name='Input' + str(i)).values # can scale it as: * 10


    xpos, ypos = np.meshgrid(np.arange(50), np.arange(50))
    zpos = data_state
    ax = plt.subplot(3,5,kk+1,projection='3d')
    ax.view_init(elev=19, azim=45)
    cmap = cm.get_cmap("jet")
    im_input = ax.plot_surface(xpos, ypos, zpos, cmap=cm.inferno, linewidth=0, antialiased=False)
    if kk == 0:
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel('State Field',fontsize=16,rotation=90)
    # ax.bar3d(xpos,ypos,zpos, dx*2, dy*2, np.zeros(2500),alpha=0.7)
    ax.set_zlim(0, 1)
    # ax.set_zlim(20, 30) # if scaling, using this range
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.grid(False)


    plt.subplot(3,5,kk+6)
    im = plt.pcolor((data_state),cmap=plt.cm.inferno)
    if kk == 0:
        plt.ylabel('State Field',fontsize=16)
    plt.clim(0,1)
    # plt.clim(20, 30) # if scaling, using this range
    plt.xticks([])
    plt.yticks([])


    ###### u
    Usource = np.zeros((71,71))
    kk_i = 0
    for ii in [16,24,32,38,46,55]:
        kk_j = 0
        for jj in [16,24,32,38,46,55]:
            # Usource[ii-1,jj-1] = data_u[kk_i,kk_j]
            Usource[ii - 1:ii+2, jj - 1:jj+2] = data_input[kk_i, kk_j]
            kk_j += 1
        kk_i += 1
    Usource = Usource[10:60,10:60]


    plt.subplot(3,5,kk+11)
    im_input = plt.pcolor((Usource),cmap=plt.cm.inferno)
    if kk == 0:
        plt.ylabel('Heat Source Input',fontsize=16)
    plt.clim(0,10)
    # plt.clim(0, 100) # if scaling, using this range
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$t$'+ ' = {} s'.format(time[kk]),fontsize=16,labelpad=13)
    kk += 1



plt.subplots_adjust(left=0.1,right=0.8,bottom=0.1,top=0.9)
bar_ax = plt.axes([0.82,0.38,0.01,0.52])
plt.xticks([])
cb = plt.colorbar(im,cax=bar_ax)
cb.ax.tick_params(labelsize=13) #13
cb.set_label("$^\circ$C",rotation=90,fontsize=13) #13
plt.show()

bar_ax2 = plt.axes([0.82,0.1,0.01,0.23])
plt.xticks([])
cb2 = plt.colorbar(im_input,cax=bar_ax2)
cb2.ax.tick_params(labelsize=13) #13
cb2.set_label("$\mathrm{W\cdot m^{-2}}$",rotation=90,fontsize=13) #13
plt.show()

