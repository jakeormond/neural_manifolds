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
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

import sklearn.metrics
import pickle

from cebra_embedding import create_folds

# sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
# from utilities.get_directories import get_data_dir


def decoding_pos_dir(emb_train, emb_test, label_train, label_test, n_neighbors=36):
    pos_decoder = KNeighborsRegressor(n_neighbors, metric = 'cosine')
    dir_decoder = KNeighborsClassifier(n_neighbors, metric = 'cosine')

    pos_decoder.fit(emb_train, label_train[:,0])
    dir_decoder.fit(emb_train, label_train[:,1])

    pos_pred = pos_decoder.predict(emb_test)
    dir_pred = dir_decoder.predict(emb_test)

    prediction =np.stack([pos_pred, dir_pred],axis = 1)

    test_score = sklearn.metrics.r2_score(label_test[:,:2], prediction)
    pos_test_err = np.median(abs(prediction[:,0] - label_test[:, 0]))
    pos_test_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:,0])

    return test_score, pos_test_err, pos_test_score


def decoding_pos(emb_train, emb_test, label_train, label_test, n_neighbors=36):
    pos_decoder = KNeighborsRegressor(n_neighbors, metric = 'cosine')

    pos_decoder.fit(emb_train, label_train)

    pos_pred = pos_decoder.predict(emb_test)

    # pos_test_err = np.median(abs(prediction - label_test[:, 0]))
    euclidean_error = np.sqrt(np.sum((pos_pred - label_test)**2, axis=1))
    pos_test_err = np.median(euclidean_error)
    
    pos_test_x_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:,0])
    pos_test_y_score = sklearn.metrics.r2_score(label_test[:, 1], prediction[:,1])

    return pos_test_err, pos_test_x_score, pos_test_y_score





def main():
    animal = 'Rat46'
    session = '19-02-2024'

    data_dir = '/ceph/scratch/jakeo/honeycomb_neural_data/rat_7/6-12-2019/'
    spike_dir = os.path.join(data_dir, 'physiology_data')
    dlc_dir = os.path.join(data_dir, 'positional_data')
    labels = np.load(f'{dlc_dir}/labels_1203_with_dist2goal_scale_data_False_zscore_data_False_overlap_False_window_size_250.npy')
    spike_data = np.load(f'{spike_dir}/inputs_overlap_False_window_size_250.npy')

    # load convert inputs to torch tensor
    inputs = torch.tensor(spike_data, dtype=torch.float32)  

    # Define the hyperparameters to tune
    param_grid = {
        'temperature': [0.11, 0.21, 1.2, 3.21],
        'time_offsets': [1, 2, 3],
        'output_dimension': [2, 3, 4, 5, 6, 7, 8],
        'batch_size': [256, 512, 1024],
        'learning_rate': [3e-5, 3e-4, 3e-3, 3e-2, 3e-2],
        # Add other hyperparameters here
    }

    # Create a grid of hyperparameter combinations
    param_combinations = list(ParameterGrid(param_grid))    

    # subsample 200 random combinations
    param_combinations = np.random.choice(param_combinations, 200, replace=False)

    
    ########### TRAIN THE CEBRA MODEL ###############
    # cebra_model_dir = os.path.join(data_dir, 'cebra')
    cebra_model_dir = data_dir
    
    max_iterations = 10000 #default is 5000.
    # max_iterations = 5000 #default is 5000.

    # will use k-folds with 10 splits
    n_splits = 10
    # kf = KFold(n_splits=n_splits, shuffle=False)
    n_timesteps = inputs.shape[0]
    num_windows = 1000
    folds = create_folds(n_timesteps, num_folds=n_splits, num_windows=num_windows)

    folds_file_name = f'folds_goal{goal}_ws{window_size}'
    folds_file_path = os.path.join(data_dir, folds_file_name + '.pkl')
    with open(folds_file_path, 'wb') as f:
        pickle.dump(folds, f)

    # for i, (train_index, test_index) in enumerate(kf.split(inputs)):
    for i, (train_index, test_index) in enumerate(folds):

        print(f'Fold {i+1} of {n_splits}')
        X_train, X_test = inputs[train_index,:], inputs[test_index,:]
        
        ################## TIME-ONLY MODEL ########################
        cebra_time3_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

        cebra_time3_model.fit(X_train)
        print('finished fitting time model')
    
        cebra_file_name = f'cebra_time3_goal{goal}_ws{window_size}_fold{i+1}.pt'
        cebra_file_path = os.path.join(cebra_model_dir, cebra_file_name)
        # need to convert any double backslashes to forward slashes      
        # cebra_file_path = cebra_file_path.replace("\\", "/")        
        cebra_time3_model.save(cebra_file_path)

        ################ SHUFFLED TIME ##################
        cebra_time_shuffled3_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

        X_train_shuffled = np.random.permutation(X_train)
        cebra_time_shuffled3_model.fit(X_train_shuffled)
        print('finished fitting time-shuffle model')

        cebra_file_name = f'cebra_time_shuffled3_goal{goal}_ws{window_size}_fold{i+1}.pt'
        cebra_file_path = os.path.join(cebra_model_dir, cebra_file_name)
        # need to convert any double backslashes to forward slashes      
        # cebra_file_path = cebra_file_path.replace("\\", "/")        
        cebra_time_shuffled3_model.save(cebra_file_path)



if __name__ == "__main__":
    main()
    

            
        
