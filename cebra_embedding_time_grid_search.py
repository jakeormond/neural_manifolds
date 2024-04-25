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
from sklearn.model_selection import KFold, ParameterGrid, PredefinedSplit, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.base import BaseEstimator

import sklearn.metrics
import pickle

from cebra_embedding import create_folds, create_folds_indicator

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


class CustomCEBRA(BaseEstimator):
    def __init__(self, model_architecture='offset10-model', batch_size=512, learning_rate=3e-4, 
                 temperature=1, output_dimension=3, max_iterations=10000, distance='cosine', 
                 conditional='time', device='cuda_if_available', verbose=True, time_offsets=10):
        
        self.model_architecture = model_architecture
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.output_dimension = output_dimension
        self.max_iterations = max_iterations
        self.distance = distance
        self.conditional = conditional
        self.device = device
        self.verbose = verbose
        self.time_offsets = time_offsets           
           

    def fit(self, X, y=None):
        self.model_ = CEBRA(model_architecture=self.model_architecture,
                            batch_size=self.batch_size,
                            learning_rate=self.learning_rate,
                            temperature=self.temperature,
                            output_dimension=self.output_dimension,
                            max_iterations=self.max_iterations,
                            distance=self.distance,
                            conditional=self.conditional,
                            device=self.device,
                            verbose=self.verbose,
                            time_offsets=self.time_offsets)

        self.model_.fit(X)
        return self

    def transform(self, X):
        return self.model_.transform(X)


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


    # # Define the CEBRA model
    cebra_model = CustomCEBRA()

    # Define the pipeline
    pipe = Pipeline(steps=[('customcebra', cebra_model)])

    # Define the hyperparameters to tune
    param_distributions = {
        'customcebra__temperature': [0.11, 0.21, 1.2, 3.21],
        'customcebra__time_offsets': [1, 2, 3],
        'customcebra__output_dimension': [2, 3, 4, 5, 6, 7, 8],
        'customcebra__batch_size': [256, 512, 1024],
        'customcebra__learning_rate': [3e-5, 3e-4, 3e-3, 3e-2, 3e-2],
        # Add other hyperparameters here
    }

    sampler = ParameterSampler(param_distributions, n_iter=200, random_state=0)

    # Create a grid of hyperparameter combinations
    # param_combinations = list(ParameterGrid(param_grid))    

    # subsample 200 random combinations
    # param_combinations = np.random.choice(param_combinations, 200, replace=False)
    
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
    # folds_file_name = f'folds_goal{goal}_ws{window_size}'
    folds_file_name = 'custom_folds'
    folds_file_path = os.path.join(data_dir, folds_file_name + '.pkl')
    with open(folds_file_path, 'wb') as f:
        pickle.dump(folds, f)

    
    for params in sampler:        

        # Set the parameters of the pipeline
        pipe.set_params(**params)

        for i, (train_index, test_index) in enumerate(folds):

            print(f'Fold {i+1} of {n_splits}')
            X_train, X_test = inputs[train_index,:], inputs[test_index,:]
            y_train, y_test = labels[train_index,:], labels[test_index,:]
            
            # Fit the model
            pipe.fit(X_train, y_train)
            print('finished fitting time model')

            # score the pipeline on the test data
            score = pipe.score(X_test, y_test)

            # print the score
            print(f'Score for fold{i}: {score}')
        
            
            



if __name__ == "__main__":
    main()
    

            
        
