# #### Fig. 8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'

# The data should be obtained from 'DMD_Training.m': save the data to the following .xlsx files
# NOTE: Each running of 'DMD_Training.m' may generate different result due to different random seeds
loc = 'transient.xlsx'
index = [2,4,7,11,16,22]
time = [2,6,10,14,18,22]

data_state = np.zeros((6,50,50))
data_input = np.zeros((6,50,50))

fig = plt.figure(32,figsize=(24,5))
kk = 0
for i in index:
    data_state = pd.read_excel(loc, header=None, sheet_name='State' + str(i)).values # can scale it as: *10 + 20
    data_input = pd.read_excel(loc, header=None, sheet_name='Input' + str(i)).values # can scale it as: *10

    plt.subplot(2,6,kk+1)
    im = plt.pcolor((data_state),cmap=plt.cm.inferno)
    if kk == 0:
        plt.ylabel('State Field',fontsize=16)
    plt.clim(0,1)
    # plt.clim(20, 30) # if scaling, using this range
    plt.xticks([])
    plt.yticks([])

    # plt.subplot(2,6,kk+7)
    # im = plt.pcolor((data_input),cmap=plt.cm.inferno)
    # if i == 0:
    #     plt.ylabel('State field (DMD)',fontsize=16)
    # plt.clim(0,10)
    # plt.xticks([])
    # plt.yticks([])

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

    plt.subplot(2,6,kk+7)
    im_input = plt.pcolor((Usource),cmap=plt.cm.inferno)
    if kk == 0:
        plt.ylabel('Heat Source Input',fontsize=16)
    plt.clim(0,10)
    # plt.clim(0, 100) # if scaling, using this range
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$t$' + ' = {} s'.format(time[kk]), fontsize=16, labelpad=13)
    kk += 1

plt.subplots_adjust(left=0.1,right=0.8,bottom=0.1,top=0.9)
bar_ax = plt.axes([0.82,0.54,0.01,0.36])
plt.xticks([])
cb = plt.colorbar(im,cax=bar_ax)
cb.ax.tick_params(labelsize=13)
cb.set_label("$^\circ$C",rotation=90,fontsize=13)
plt.show()

bar_ax2 = plt.axes([0.82,0.1,0.01,0.36])
plt.xticks([])
cb2 = plt.colorbar(im_input,cax=bar_ax2)
cb2.ax.tick_params(labelsize=13)
cb2.set_label("$\mathrm{W\cdot m^{-2}}$",rotation=90,fontsize=13)
plt.show()
