# #### Fig. 5
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'

# The data should be obtained from 'DMD_Training.m': save the data to the following .xlsx files
# NOTE: Each running of 'DMD_Training.m' may generate different result due to different random seeds
loc_true = 'data_true.xlsx'
loc_pred = 'data_pred.xlsx'
loc_u = 'data_u.xlsx'

data_true = np.zeros((5,50,50))
data_pred = np.zeros((5,50,50))
data_u = np.zeros((6,6))

fig = plt.figure(32,figsize=(16,8))
for i in range(5):
    data_true[i] = pd.read_excel(loc_true,header=None,sheet_name='Sheet'+str(i+1)).values # can scale it by: *10 + 20
    data_pred[i] = pd.read_excel(loc_pred, header=None, sheet_name='Sheet' + str(i + 1)).values# can scale it by:*10 + 20
    data_u = pd.read_excel(loc_u, header=None, sheet_name='Sheet' + str(i + 1)).values# can scale it by:*10

    plt.subplot(3,5,i+1)
    im = plt.pcolor((data_true[i]),cmap=plt.cm.inferno)
    if i == 0:
        plt.ylabel('State Field (True)',fontsize=16)
    plt.clim(-0.2,0.2)
    # plt.clim(18, 22) # if scaling, using this range
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3,5,i+6)
    im = plt.pcolor((data_pred[i]),cmap=plt.cm.inferno)
    if i == 0:
        plt.ylabel('State Field (DMD)',fontsize=16)
    plt.clim(-0.2,0.2)
    # plt.clim(18, 22) # if scaling, using this range
    plt.xticks([])
    plt.yticks([])


    ###### u
    Usource = np.zeros((71,71))
    kk_i = 0
    for ii in [16,24,32,38,46,55]:
        kk_j = 0
        for jj in [16,24,32,38,46,55]:
            # Usource[ii-1,jj-1] = data_u[kk_i,kk_j]
            Usource[ii - 1:ii+2, jj - 1:jj+2] = data_u[kk_i, kk_j]
            kk_j += 1
        kk_i += 1
    Usource = Usource[10:60,10:60]

    plt.subplot(3,5,i+11)
    im_input = plt.pcolor((Usource),cmap=plt.cm.inferno)
    if i == 0:
        plt.ylabel('Heat Source Input',fontsize=16)
    plt.clim(-4.5,4.5)
    # plt.clim(-45, 45) # if scaling, using this range
    plt.xticks([])
    plt.yticks([])



plt.subplots_adjust(left=0.1,right=0.8,bottom=0.1,top=0.9)
bar_ax = plt.axes([0.82,0.38,0.01,0.52])
plt.xticks([])
cb = plt.colorbar(im,cax=bar_ax)
cb.ax.tick_params(labelsize=14)
cb.set_label("$^\circ$C",rotation=90,fontsize=13)
plt.show()

bar_ax2 = plt.axes([0.82,0.09,0.01,0.25])
plt.xticks([])
cb2 = plt.colorbar(im_input,cax=bar_ax2)
cb2.ax.tick_params(labelsize=14)
cb2.set_label("$\mathrm{W\cdot m^{-2}}$",rotation=90,fontsize=13)
plt.show()