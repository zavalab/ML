# -*- coding: utf-8 -*-
"""
deep_cnn.py
The purpose of this code is to run deep_cnn training and output the weights into 
a pickle file. Once this is completed, we evaluate the deep CNN with recall/precision plots, etc.

Created on: 03/15/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)
    - ADD USERS HERE

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
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

## IMPORTING SKLEARN MODULES
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


### FUNCTION TO LOCATE TEST INSTANCE VALUE
def locate_test_instance_value( csv_file, solute, cosolvent, mass_frac_water  ):
    '''
    This code locates the test instance class value.
    INPUTS:
        csv_file: [object]
            csv file
        solute: [str]
            name of the solute
        cosolvent: [str]
            name of the cosolvent
        mass_frac_water: [str]
            mass fraction of water
    OUTPUTS:
        value: [str]
            positive or negative for class value
    '''
    location = csv_file.loc[(csv_file['solute'] == solute) & (csv_file['cosolvent'] == cosolvent) & (csv_file['mass_frac_water'] == int(mass_frac_water)), 'sigma_label']
    ## CONVERTING LOCATION TO A STRING
    value = list(location)[0]
    return value

### FUNCTION TO CALL THE 3D COVOLUTIONAL NEURAL NETWORK
def cnn():
    ## INPUT LAYER
    input_layer = Input((20, 20, 20, 3))
    
    ## CONVOLUTIONAL LAYERS
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)
    
    ## MAXPOOLING LAYER
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)
    
    ## CONVOLUTIONAL LAYERS
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
    conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
    
    ## MAXPOOLING LAYER
    pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)
    
    ## BATCH NORMALIZATION ON THE CONVOLUTIONAL OUTPUTS BEFORE FULLY CONNECTED LAYERS
    pooling_layer2 = BatchNormalization()(pooling_layer2)
    flatten_layer = Flatten()(pooling_layer2)
    
    ## FULLY CONNECTED LAYERS/ DROPOUT TO PREVENT OVERFITTING
    dense_layer1 = Dense(units=4096, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(1)(dense_layer1)
    dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(1)(dense_layer2)
    output_layer = Dense(units=2, activation='softmax')(dense_layer2)
    
    ## DEFINE MODEL WITH INPUT AND OUTPUT LAYERS
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

## STORE X DATA AND ITS LABEL (POSITIVE OR NEGATIVE)
x_data = []
y_label = []
#%%
## MAIN FUNCTION
if __name__ == "__main__":
    np.random.seed(0)
    ## PART I: DATA COLLECTION
    ## SOLUTE
    for i in list(SOLUTE_TO_TEMP_DICT.keys()):
        ## COSOLVENT (IF WANT TO COMBINE ALL, PUT EVERY COSOLVENT INTO THE LIST)
        for j in ['DIO']:
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
                # database_path=r"C:\Users\jsl95\OneDrive - UW-Madison\CS 760\Gridded_Data\Gridded_Data"
                database_path = r"C:\Users\akchew\Box Sync\2019_Spring\CS760\Spring_2019_CS760_Project\Datasets\Gridded_Data"
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
                if (class_instance_value == 'positive'):
                    y_label.append(1)
                elif (class_instance_value == 'negative'):
                    y_label.append(0)
            
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
                ensemble_avg = np.average(training_data_for_instance[800:1001,:,:,:,:], axis = 0)
                
                if (class_instance_value == 'positive' or class_instance_value == 'negative'):
                    x_data.append(ensemble_avg)
                ## PLOTTING TRAINING DATA
                #fig, ax = plot_voxel(training_data_for_instance, 1000) # Outputs an xyz plot
    
    ## PART II: PREPROCESS FEATURES AND LABELS
    ## CHANGE TO ARRAY FORMAT FOR X
    x = np.asarray(x_data)
    
    ## ONE-HOT ENCODING Y
    y = to_categorical(y_label)
    
    ## SPLIT DATA
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
    #%%
    ## CALL 3D-CNN
    model = cnn()
    
    ## COMPILE CNN AND TRAIN
    '''
    Loss function is categorical_crossentropy
    Optimizer is adam with lr = 0.00001
    Metrics is accuracy
    A check point is set as finding the weights with max validation accuracy and store it
    A check point can also be min validation loss
    '''
    trained = 'no' # or 'yes'
    ## CHANGE TO YOUR DIRECTORY TO STORE WEIGHTS
    filepath=database_path+"\weightsDIO.best.hdf5"
    
    if trained == 'no':
        model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.00001), metrics=['acc'])
        #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        model.fit(x=x_train, y=y_train, batch_size=18, epochs=500, validation_split=0.2, callbacks=callbacks_list)
    
    ## TEST CNN
    model.load_weights(filepath)
    
    # COMPILE MODEL AGAIN WITH STORED WEIGHTS
    model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.00001), metrics=['acc'])
    model.summary()
    print("Created model and loaded weights from file")
    
    
    ## ACCURACY METRICS
    pred = model.predict(x_test)
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
        
    accuracy_score(y_true, y_pred)