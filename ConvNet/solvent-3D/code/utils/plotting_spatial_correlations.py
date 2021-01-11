# -*- coding: utf-8 -*-
"""
plotting_spatial_correlations.py
The purpose of this code is to load the SolventNet model and the instances, then 
generate spatial correlation maps that could give insight into what the 3D CNN 
is doing. This code was designed for Shengli (Bruce) Jiang to make maps. 

Created on: 02/10/2020

"""
## IMPORTING OS COMMANDS
import os

## TRAINING TOOLS
from train_deep_cnn import split_train_test_set

## ANALYSIS TOOLS
from analyze_deep_cnn import find_avg_std_predictions, create_dataframe

## PICKLE FUNCTIONS
from extraction_scripts import load_pickle_general

## LOADING THE MODELS
from keras.models import load_model

## LOADING NOMENCLATURE
from core.nomenclature import read_combined_name, extract_sampling_inputs, extract_instance_names

## PLOTTING PARITY PLOT
from read_extract_deep_cnn import plot_parity_publication_single_solvent_system


#%%
## MAIN FUNCTION
if __name__ == "__main__":
    ## DEFINING PATH TO SIMULATION FOLDER
    sim_folder=r"/home/sjiang87/machinelearning2"
    
    ## DEFINING SPECIFIC SIM
    specific_sim = r"3DCNN_Alex"
    
    ## DEFINING SOLVENT NET SIM FOLDER
    training_folder=r"20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-strlearn-0.80-solvent_net-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF"
    
    ## DEFINING INSTANCES FILE
    instances_file = r"20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-DIO_GVL_THF-10_25_50_75"
    
    ## DEFINING PATHS
    path_to_sim = os.path.join(sim_folder,
                               specific_sim)
    path_training_folder = os.path.join(path_to_sim,
                                        training_folder)
    
    ## DEFINING PATH TO INSTANCES
    path_instances = os.path.join(path_to_sim,
                                  instances_file)
    
    ## DEFINING THE MODELS 
    model_list = [
            'model_fold_0.hdf5',
            'model_fold_1.hdf5',
            'model_fold_2.hdf5',
            'model_fold_3.hdf5',
            'model_fold_4.hdf5',
            ]
    
    #%% LOADING ALL INSTANCES
    instances = load_pickle_general(path_instances)
    '''
    instances outputs everything of shape 3:
        Index 0: voxel representations in 76 x 10 x (20 x 20 x 20 x 3)
            where each reactant/water has 10 partitions (i.e. 10 voxel representations)
            The last 2 voxel representations are used for testing
        Index 1: Experimental sigma values in with length of 76, e.g.
            [1.15,
             0.84,
             0.21,
             0.05,
             1.6,
             0.72,
             ...
             ]
            
        Index 2: Labels for each, e.g.
            ['CEL_403.15_DIO_10',
             'CEL_403.15_DIO_25',
             'CEL_403.15_DIO_50',
             'CEL_403.15_DIO_75',
             ...
             ]
            Where 'CEL_403.15_DIO_10' means cellobiose simulated at 403.15 K, with 10 wt% water / 90 wt% dioxane
    '''
    
    #%% LOADING THE TRAINED MODELS
    
    ## DEFINING SPECIFIC MODEL
    model_index = 0
    
    ## LOADING SPECIFIC MODEL INDEX
    path_model = os.path.join(path_training_folder,
                              model_list[model_index])
    
    ## LOADING MODEL
    model = load_model(path_model)
    ## PRINT MODEL SUMMARY
    model.summary()
    '''
    This should output the summary of solvent net, e.g.
        Layer (type)                 Output Shape              Param #   
        =================================================================
        input_1 (InputLayer)         (None, 20, 20, 20, 3)     0         
        _________________________________________________________________
        conv3d_1 (Conv3D)            (None, 18, 18, 18, 8)     656       
        _________________________________________________________________
        conv3d_2 (Conv3D)            (None, 16, 16, 16, 16)    3472      
        _________________________________________________________________
        ...
        dense_3 (Dense)              (None, 128)               16512     
        _________________________________________________________________
        dropout_3 (Dropout)          (None, 128)               0         
        _________________________________________________________________
        dense_4 (Dense)              (None, 1)                 129       
        =================================================================
        Total params: 172,417
        Trainable params: 172,289
        Non-trainable params: 128
        _________________________________________________________________
        
    '''
    
    #%% PREDICTING WITH THE TRAINED MODEL
    
    ## GETTING NAME INFORMATION
    combined_name_info = read_combined_name(training_folder)
    '''
    Outputs a dictionary with the training information
        {'data_type': '20_20_20_20ns_oxy_3chan',            # 20 x 20 x 20 channel with oxygen as a reactant channel
         'representation_type': 'split_avg_nonorm',         # split trajectory
         'representation_inputs': '10',                     # split trajectory 10 ways (for a 20 ns simulation)
         'sampling_type': 'strlearn',                       # Use stratified learning to divide training and test set
         'sampling_inputs': '0.80',                         # Divide training and test set by 80%
         'cnn_type': 'solvent_net',                         # SolventNet was used
         'epochs': '500',                                   # 500 epoches was trained for
         'solute_list': ['CEL', 'ETBE', 'FRU', 'LGA', 'PDO', 'XYL', 'tBuOH'], # solutes that were within training set
         'mass_frac_data': ['10', '25', '50', '75'],                          # mass fractions that were withing training set  
         'solvent_list': ['DIO', 'GVL', 'THF'],                               # solvents that were in training set
         'want_descriptor': False}                                            # whether descriptors were added to the last layer of SolventNet
    '''
    
    ## GENERATING SAMPLING DICT
    sampling_dict = extract_sampling_inputs( sampling_type = combined_name_info['sampling_type'], 
                                             sampling_inputs = [ combined_name_info['sampling_inputs'] ])
    
    ## GETTTING TRAINING AND TESTING DATA
    x_train, x_test, y_train, y_test = split_train_test_set( sampling_dict = sampling_dict,
                                                             x_data = instances[0],
                                                             y_label = instances[1])
    
    #%%
    ## PREDICTIONS
    y_pred = model.predict(x_test).reshape(len(y_test) )
    ## OUTPUTS PREDICTIONS FOR 76 * 2 voxels = 156 TOTAL PREDICTIONS
    
    ## DEFINING INSTANCE NAMES
    instance_names = instances[2]
    ''' List of instance names, e.g.
        ['CEL_403.15_DIO_10',
         'CEL_403.15_DIO_25',
         'CEL_403.15_DIO_50',
         ...
         ]
    '''
    
    ## GETTING ERROR BARS
    y_pred_avg, y_pred_std, y_true_split = find_avg_std_predictions(instance_names = instance_names,
                                                                    y_pred = y_pred,
                                                                    y_true = y_test)
    
    ## GETTING INSTANCE DICTIONARY
    instance_dict = [ extract_instance_names(name = name) for name in instance_names ]
    ''' Generates dictionary with instances
        [{'solute': 'CEL', 'temp': '403.15', 'cosolvent': 'DIO', 'mass_frac': '10'},
         {'solute': 'CEL', 'temp': '403.15', 'cosolvent': 'DIO', 'mass_frac': '25'}]
    '''
    
    ## GENERATING DATAFRAME
    dataframe = create_dataframe(instance_dict = instance_dict,
                                 y_true = y_true_split,
                                 y_pred = y_pred_avg,
                                 y_pred_std = y_pred_std)
    
    ## DEFINING FIGURE SIZE
    figure_size=( 18.542/2, 18.542/2 ) # in cm
    
    ## PLOTTING PARITY
    plot_parity_publication_single_solvent_system( dataframe = dataframe,
                                                   fig_name =  os.path.join(path_to_sim, "solventnet_model_%d_performance.png"%(model_index) ),
                                                   mass_frac_water_label = 'mass_frac',
                                                   sigma_act_label = 'y_true',
                                                   sigma_pred_label = 'y_pred',
                                                   sigma_pred_err_label = 'y_pred_std',
                                                   fig_extension = 'png',
                                                   save_fig_size = figure_size,
                                                   save_fig = True)
    
    
#%%
#%% Saliency tubes
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from vis.visualization import visualize_saliency, overlay, visualize_cam
titles = ['right steering', 'left steering', 'maintain steering']
modifiers = [None, 'negate', 'small_values']

vid = x_test
def grad(idxx):
    img = vid[idxx,:,:,:,:]
    layer_input = model.input
    loss = model.layers[-1].output[...,0]
    grad_tensor = K.gradients(loss,layer_input)[0]
    derivative_fn = K.function([layer_input],[grad_tensor])
    grad_eval_by_hand = derivative_fn([img[np.newaxis,...]])[0]
    grad_eval_by_hand = np.abs(grad_eval_by_hand).max(axis=(0,-1))
    arr_min, arr_max  = np.min(grad_eval_by_hand), np.max(grad_eval_by_hand)
    grad_eval_by_hand = (grad_eval_by_hand - arr_min) / (arr_max - arr_min + K.epsilon())
    return img, grad_eval_by_hand, y_test

def grad2(idxx):
    img = vid[idxx,:,:,:,:]
    grad_eval_by_hand = visualize_saliency(model, layer_idx=-1, filter_indices=0, 
                            seed_input=img, grad_modifier='negate', keepdims=True)
    grad_eval_by_hand2 = visualize_saliency(model, layer_idx=-1, filter_indices=0, 
                            seed_input=img, grad_modifier=None, keepdims=True)
    grad_eval_by_hand = grad_eval_by_hand2
    return img, grad_eval_by_hand, y_test
    
    
img, grad_eval_by_hand, y_test = grad2(0)
fig, axs = plt.subplots(8,5, figsize=(18, 12), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.3, wspace=.001)


axs = axs.ravel()
maxx = np.max(np.sum(grad_eval_by_hand, axis=(1,2)))
for i in range(20):
    axs[i].imshow(grad_eval_by_hand[i,:,:,0])
    axs[i+20].imshow(img[i,:,:,:], vmin=0, vmax=1)
    axs[i].set_title('{} s/ {:.1f}'.format(i, np.sum(grad_eval_by_hand[i,:,:])/maxx))
    axs[i].axis('off')
plt.tight_layout()
#%%
img, grad_eval_by_hand, y_test = grad2(26)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
fit = lambda x: scaler.fit_transform(x)
sal = np.sum(grad_eval_by_hand[:,:,:,0], axis=(1,2))[...,np.newaxis]
water = np.sum(img[:,:,:,0], axis=(1,2))[...,np.newaxis]
reactant = np.sum(img[:,:,:,1], axis=(1,2))[...,np.newaxis]
cosolvent = np.sum(img[:,:,:,2], axis=(1,2))[...,np.newaxis]

plt.plot(fit(sal), '-o', color='r', label='importance max {:.0f}'.format(sal.max()))
plt.plot(fit(water), '--d', color='black', label='water max {:.0f}'.format(water.max()))
plt.plot(fit(reactant), '--*', color = 'dimgray', label='reactant max {:.0f}'.format(reactant.max()))
plt.plot(fit(cosolvent), '--h', color = 'silver', label='cosolvent max {:.0f}'.format(cosolvent.max()))

plt.grid()
plt.legend()
plt.tight_layout()
