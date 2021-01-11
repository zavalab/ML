# -*- coding: utf-8 -*-
"""
deep_cnn_vox_net.py

Created on: 04/15/2019

@author: 
    - Shengli Jiang (sjiang87@wisc.edu)
"""
## IMPORTING NECESSARY MODULES
import os

## IMPORTING PANDAS
import pandas as pd

## IMPORTING NUMPY
import numpy as np

## IMPORTING GLOBAL VARIABLES
from core.global_vars import SOLUTE_TO_TEMP_DICT

## TAKING EXTRACTION SCRIPTS
from extraction_scripts import load_pickle

## TAKING NOMENCLATURE
from core.nomenclature import convert_to_single_name

## PLOTTING MODULES
from core.plotting_scripts import plot_voxel

## IMPORTING KERAS MODULES
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

## IMPORTING SKLEARN MODULES
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

## IMPORT MATPLOTLIB
import matplotlib.pyplot as plt

## IMPORTING FUNCTION FOR TEST INSTANCE VALUE
from core.ml_funcs import locate_test_instance_value

## FUNCTION TO CALCUALTE MSE, R2, EVS, MAE
def metrics(y_fit,y_act):
    evs = explained_variance_score(y_act, y_fit)
    mae = mean_absolute_error(y_act, y_fit)
    mse = mean_squared_error(y_act, y_fit)
    r2 = r2_score(y_act, y_fit)
    return mae, mse, evs, r2

## VOXNET
def vox_cnn(input_data_shape, regress=True):
    '''
    Model for VoxNet. 
    INPUTS:
        input_data_shape: [tuple]
            input data shape
        regress: [logical, default=True]
            True if you want your model to have a linear regression at the end
    OUTPUT:
        model: [obj]
            tensorflow model
    '''
    ## INPUT LAYER
    input_layer = Input(input_data_shape)
    
    ## CONVOLUTIONAL LAYERS
    conv_layer1 = Conv3D(filters=32, kernel_size=(5, 5, 5), strides=(2, 2, 2), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(conv_layer1)
    
    ## MAXPOOLING LAYER
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)
    
    ## BATCH NORMALIZATION ON THE CONVOLUTIONAL OUTPUTS BEFORE FULLY CONNECTED LAYERS
    pooling_layer1 = BatchNormalization()(pooling_layer1)
    flatten_layer = Flatten()(pooling_layer1)
    
    ## FULLY CONNECTED LAYERS/ DROPOUT TO PREVENT OVERFITTING
    dense_layer1 = Dense(units=128, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(1)(dense_layer1)
    #output_layer = Dense(units=2, activation='softmax')(dense_layer1)
    if regress is True:
        ## CREATING LINEAR MODEL
        output_layer = Dense(units=1, activation='linear')(dense_layer1)
    else:
        ## NO OUTER LAYER, JUST DENSE LAYER
        output_layer = dense_layer1
    
    ## DEFINE MODEL WITH INPUT AND OUTPUT LAYERS
    model = Model(inputs=input_layer, outputs=output_layer)
    return model



## MAIN FUNCTION
if __name__ == "__main__":
    np.random.seed(0)
    '''PART I: DATA COLLECTION'''
    
    ## STORE X DATA AND ITS LABEL (SIGMA)
    x_data = []
    y_label = []
    y_solvent = []
    ## SOLUTE
    for i in list(SOLUTE_TO_TEMP_DICT.keys()):
        ## COSOLVENT (IF WANT TO COMBINE ALL, PUT EVERY COSOLVENT INTO THE LIST)
        for j in ['DIO','THF','GVL']:
            ## MASS FRACTION OF WATER
            for k in ['10', '25', '50', '75']:
                ## DEFINING SPECIFIC TRAINING INSTANCE
                training_instance = {
                        'solute': i,
                        'cosolvent': j,
                        'mass_frac': k, # mass fraction of water
                        'temp': SOLUTE_TO_TEMP_DICT[i], # Temperature
                        }
                ## INCLUDING TEMPERATURE
                training_instance['temp'] = SOLUTE_TO_TEMP_DICT[training_instance['solute']]
            
                ## DEFINING FULL PATH TO YOUR DATABASE (CHANGE TO YOUR OWN DIRECTORY)
                database_path=r"C:\Users\jsl95\OneDrive - UW-Madison\Course\CS 760\Gridded_Data\Gridded_Data"
                
                ## DEFINING CLASS CSV FILE
                class_file_name="solvent_effects_classification_data.csv"
                
                ## DEFINING FULL PATH
                class_file_path = os.path.join(database_path, class_file_name)
                
                ## READING CSV FILE
                csv_file = pd.read_csv( class_file_path )
                
                ## EXTRACTING CLASS VALUE
                class_instance_value = locate_test_instance_value(
                                                                    csv_file = csv_file,
                                                                    solute =  training_instance['solute'],
                                                                    cosolvent = training_instance['cosolvent'],
                                                                    mass_frac_water = training_instance['mass_frac'],
                                                                    )
                ## PRINTING CLASS VALUE
                print("Class value for this training instance is: %s"%( class_instance_value ) ) # Should output negative
#                if (class_instance_value == 'positive'):
#                    y_label.append(1)
#                elif (class_instance_value == 'negative'):
#                    y_label.append(0)
                if (str(class_instance_value) != 'nan'):
                    y_label.append(class_instance_value)
                    y_label.append(class_instance_value)
                    y_label.append(class_instance_value)
                    y_label.append(class_instance_value)
                    y_label.append(class_instance_value)
            
                ## CONVERTING TRAINING INSTANCE NAME TO NOMENCLATURE
                training_instance_name = convert_to_single_name( 
                                                                solute = training_instance['solute'],
                                                                solvent = training_instance['cosolvent'],
                                                                mass_fraction = training_instance['mass_frac'],
                                                                temp = training_instance['temp']
                                                                )
                
                ## DEFINING FULL TRAINING PATH
                full_train_pickle_path= os.path.join(database_path, training_instance_name)
                
                ## EXTRACTION PROTOCOL A PARTICULAR TRAINING EXAMPLE
                training_data_for_instance = load_pickle(full_train_pickle_path)
                print(training_data_for_instance.shape) # OUTPUT: (1001, 20, 20, 20, 3)
                
                ## I TAKE THE AVG, MANY OTHER WAYS ARE AVAILABLE
                ensemble_avg1 = np.average(training_data_for_instance[0:200,:,:,:,:], axis = 0)
                ensemble_avg2 = np.average(training_data_for_instance[201:400,:,:,:,:], axis = 0)
                ensemble_avg3 = np.average(training_data_for_instance[401:600,:,:,:,:], axis = 0)
                ensemble_avg4 = np.average(training_data_for_instance[601:800,:,:,:,:], axis = 0)
                ensemble_avg5 = np.average(training_data_for_instance[801:1001,:,:,:,:], axis = 0)
                
                if (str(class_instance_value) != 'nan'):
                    x_data.append(ensemble_avg1)
                    x_data.append(ensemble_avg2)
                    x_data.append(ensemble_avg3)
                    x_data.append(ensemble_avg4)
                    x_data.append(ensemble_avg5)
                    y_solvent.append(j)
                    
    '''PART II: PREPROCESS FEATURES AND LABELS'''
    ## CHANGE TO ARRAY FORMAT FOR X
    x = np.asarray(x_data)
    
    #y = to_categorical(y_label)
    y = y_label
    
    ## SPLIT DATA
    y = pd.Series(y)
    x_train = np.empty((0,20,20,20,3))
    y_train = pd.Series()
    x_test = np.empty((0,20,20,20,3))
    y_test = pd.Series()
    
    ## STRATIFY THE DATA
    '''Every five data points have first three as training and last two as testing'''
    for i in range(3):
        x_train = np.concatenate((x_train, x[i::5,:,:,:,:]), axis=0)
        y_train = y_train.append(y[i::5])
        
    for i in range(3,5):
        x_test = np.concatenate((x_test, x[i::5,:,:,:,:]), axis=0)
        y_test = y_test.append(y[i::5])    

            
    '''PART III: TRAIN THE MODEL'''
    y_index = np.array(y_test.index)
    y_train = y_train.values
    y_test = y_test.values
    
    model = vox_cnn()
    
    ## COMPILE CNN AND TRAIN
    '''
    Loss function is MSE
    Optimizer is adam with lr = 0.00001
    Metrics is accuracy
    A check point is set as finding the weights with min MSE and store it
    A check point can also be min validation loss
    '''
    trained = 'yes' # or 'no' if 'no' it will retrain the data
    ## CHANGE TO YOUR DIRECTORY TO STORE WEIGHTS
    filepath= r"C:\Users\jsl95\OneDrive - UW-Madison\Course\CS 760\Gridded_Data\Gridded_Data"+"\weights_stratified.best.hdf5"
    
    if trained == 'no':
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=0.00001), metrics=['mean_squared_error'])
        #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = model.fit(x=x_train, y=y_train, batch_size=18, epochs=500, validation_split=0.2, callbacks=callbacks_list)
    
    ## TEST CNN
    model.load_weights(filepath)
    
    # COMPILE MODEL AGAIN WITH STORED WEIGHTS
    model.compile(loss=mean_squared_error, optimizer=Adam(lr=0.00001), metrics=['mean_squared_error'])
    model.summary()
    print("Created model and loaded weights from file")


    '''PART IV: DATA ANALYSIS'''
    ## PLOT LEARNING CURVE
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model mean squared error')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    ## ACCURACY METRICS
    y_test_solvent_index = np.divmod(y_index, 5)[0]
    pred = model.predict(x_test)
    y_pred = pred.reshape(len(y_test))
    y_true = np.asarray(y_test)
   
    DIO_idx = []
    THF_idx = []
    GVL_idx = []
    for i in range(len(y_test)):
        idx = y_test_solvent_index[i]
        solvent_name = y_solvent[idx]
        if solvent_name == 'DIO':
            DIO_idx.append(i)
        if solvent_name == 'THF':
            THF_idx.append(i)
        if solvent_name == 'GVL':
            GVL_idx.append(i)
    DIO_true = y_true[DIO_idx]
    DIO_pred = y_pred[DIO_idx] 
    THF_true = y_true[THF_idx]
    THF_pred = y_pred[THF_idx] 
    GVL_true = y_true[GVL_idx]
    GVL_pred = y_pred[GVL_idx] 
    
    plt.rc('font', family = 'serif')
    fig = plt.figure(figsize=(6,6))        
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(DIO_true, DIO_pred, c='g')
    ax.scatter(THF_true, THF_pred, c='b')
    ax.scatter(GVL_true, GVL_pred, c='r')
    ax.plot(np.linspace(-1.2,2.2), np.linspace(-1.2,2.2), 'r')
    ax.set_xlabel('Actual sigma',fontsize=15)
    ax.set_ylabel('Predicted sigma',fontsize=15)
    ax.grid()
    ax.set_axisbelow(True)
    ax.tick_params(direction = 'in')
    plt.tight_layout()
    plt.legend(['','DIO', 'THF', 'GVL'])
            
    from sklearn.metrics import mean_squared_error
    print('DIO mae, mse, evs, r2: \n', *metrics(DIO_true, DIO_pred), sep ='\n')
    print('THF mae, mse, evs, r2: \n', *metrics(THF_true, THF_pred), sep ='\n')
    print('GVL mae, mse, evs, r2: \n', *metrics(GVL_true, GVL_pred), sep ='\n')
