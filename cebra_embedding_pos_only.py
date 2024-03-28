import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.collections import LineCollection
import os
# import cebra.datasets
import cebra
import torch
from cebra import CEBRA
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
import pickle

from cebra_embedding import create_folds

# sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
# from utilities.get_directories import get_data_dir


''' using data created with pytorch_decoding.dataset_creation.py '''

if __name__ == "__main__":
    animal = 'Rat46'
    session = '19-02-2024'
    # data_dir = get_data_dir(animal, session)

    data_dir = '/ceph/scratch/jakeo/'

    goal = 52
    window_sizes = [100, 250]

    for window_size in window_sizes: 


        ############ LOAD POSITIONAL DATA ################
        #  = os.path.join(data_dir, 'deeplabcut')
        labels_file_name = f'labels_goal{goal}_ws{window_size}'
        # load numpy array of labels
        labels = np.load(os.path.join(data_dir, labels_file_name + '.npy'))
        # keep only the first 2 columns
        labels = labels[:, :2]

        ############ LOAD SPIKE DATA #####################
        # load numpy array of neural data
        # spike_dir = os.path.join(data_dir, 'spike_sorting')
        inputs_file_name = f'inputs_goal{goal}_ws{window_size}'
        inputs = np.load(os.path.join(data_dir, inputs_file_name + '.npy'))

        # load convert inputs to torch tensor
        inputs = torch.tensor(inputs, dtype=torch.float32)  

        ########### TRAIN THE CEBRA MODEL ###############
        # cebra_model_dir = os.path.join(data_dir, 'cebra')
        cebra_model_dir = data_dir
        
        # max_iterations = 10000 #default is 5000.
        max_iterations = 5000 #default is 5000.

        # will use k-folds with 5 splits
        n_splits = 10
        # kf = KFold(n_splits=n_splits, shuffle=False)
        n_timesteps = inputs.shape[0]
        num_windows = 200
        folds = create_folds(n_timesteps, num_folds=n_splits, num_windows=num_windows)
        
        folds_file_name = f'folds_goal{goal}_ws{window_size}'
        folds_file_path = os.path.join(data_dir, folds_file_name + '.pkl')
        with open(folds_file_path, 'wb') as f:
            pickle.dump(folds, f)


        # folds_v2 = create_folds_v2(n_timesteps, num_folds=n_splits, num_windows=10)
        # folds_v2_file_name = f'folds_v2_goal{goal}_ws{window_size}'
        # folds_v2_file_path = os.path.join(data_dir, folds_v2_file_name + '.pkl')
        # with open(folds_v2_file_path, 'wb') as f:
        #     pickle.dump(folds_v2, f)

        # for i, (train_index, test_index) in enumerate(kf.split(inputs)):
        for i, (train_index, test_index) in enumerate(folds):

            print(f'Fold {i+1} of {n_splits}')
            X_train, X_test = inputs[train_index,:], inputs[test_index,:]
            y_train, y_test = labels[train_index,:], labels[test_index,:]

            ################ BEHAVIOUR MODEL WITH POSITION #######################
            cebra_pos3_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=1,
                            output_dimension=3,
                            max_iterations=max_iterations,
                            distance='cosine',
                            conditional='time_delta',
                            device='cuda_if_available',
                            verbose=True,
                            time_offsets=10)

            cebra_pos3_model.fit(X_train, y_train)
            print('finished fitting position model')
        
            cebra_file_name = f'cebra_pos3_goal{goal}_ws{window_size}_fold{i+1}.pt'
            cebra_file_path = os.path.join(cebra_model_dir, cebra_file_name)
            # need to convert any double backslashes to forward slashes      
            # cebra_file_path = cebra_file_path.replace("\\", "/")        
            cebra_pos3_model.save(cebra_file_path)

            #####################################################
            # train_index, test_index = folds_v2[i]

            # print(f'Fold {i+1} of {n_splits}')
            # X_train, X_test = inputs[train_index,:], inputs[test_index,:]
            # y_train, y_test = labels[train_index,:], labels[test_index,:]

            ################ BEHAVIOUR MODEL WITH POSITION #######################
            # cebra_pos3_model_v2 = CEBRA(model_architecture='offset10-model',
            #                 batch_size=512,
            #                 learning_rate=3e-4,
            #                 temperature=1,
            #                 output_dimension=3,
            #                 max_iterations=max_iterations,
            #                 distance='cosine',
            #                 conditional='time_delta',
            #                 device='cuda_if_available',
            #                 verbose=True,
            #                 time_offsets=10)

            # cebra_pos3_model_v2.fit(X_train, y_train)
            # print('finished fitting position model')
        
            # cebra_file_name = f'cebra_pos3_v2_goal{goal}_ws{window_size}_fold{i+1}.pt'
            # cebra_file_path = os.path.join(cebra_model_dir, cebra_file_name)
            # # need to convert any double backslashes to forward slashes      
            # # cebra_file_path = cebra_file_path.replace("\\", "/")        
            # cebra_pos3_model.save(cebra_file_path)

            
        
