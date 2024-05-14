import sys
# import logging
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
# from matplotlib.collections import LineCollection
import os
# import cebra.datasets
import cebra
import torch
from cebra import CEBRA
from sklearn.model_selection import KFold, ParameterGrid, PredefinedSplit, ParameterSampler, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.base import BaseEstimator
import scipy.ndimage
from sklearn.metrics import make_scorer, r2_score
from skopt import BayesSearchCV
import pickle

from cebra_embedding import create_folds

# sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
# from utilities.get_directories import get_data_dir


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


class Logger:
    
    def __init__(self, log_file):
        # self.iteration_number = 0 # can only count iterations if running search serially, not in parallel
        self.log_file = log_file

    def log_and_score(self, y_true, y_pred):
        
        score = r2_score(y_true, y_pred)
        with open(self.log_file, 'a') as f:
            f.write(f'Finished iteration with score: {score}\n')
        return score


def main():    
        
    # check if pytorch has access to GPU
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    animal = 'rat_9'
    session = '10-12-2021'

    data_dir = os.path.join('/ceph/scratch/jakeo/honeycomb_neural_data/', animal, session)
    
    # create model directory
    model_dir = os.path.join('/ceph/scratch/jakeo/honeycomb_neural_data/models/', animal)
    # model_dir = '/ceph/scratch/jakeo/honeycomb_neural_data/models/'

    dlc_dir = os.path.join(data_dir, 'positional_data')
    labels = np.load(f'{dlc_dir}/labels_250.npy')
    # keep the first 2 columns and the last 2 columns
    labels = labels[:, [0, 1, -2, -1]]

    spike_dir = os.path.join(data_dir, 'physiology_data')
    spike_data = np.load(f'{spike_dir}/inputs_zscored_250.npy')

    # convert inputs to torch tensor
    inputs = torch.tensor(spike_data, dtype=torch.float32)  

    # Define the CEBRA model
    cebra_model = CustomCEBRA()

    # Define the knn regressor
    knn = KNeighborsRegressor()

    # Define the pipeline
    pipe = Pipeline(steps=[('customcebra', cebra_model), ('knn', knn)])

    # Define the hyperparameters to tune
    # 'customcebra__output_dimension': [2, 3, 4, 5, 6, 7, 8]
    # param_distributions = {
    #    'customcebra__temperature': [0.11, 0.21, 1.2, 3.21],
    #    'customcebra__time_offsets': [1, 2, 3],
    #    'customcebra__output_dimension': [3, 4, 5, 6, 7, 8, 9],
    #    'customcebra__batch_size': [256, 512, 1024],
    #    'customcebra__learning_rate': [3e-5, 3e-4, 3e-3, 3e-2, 3e-2],
    #    'knn__n_neighbors': [2, 5, 10, 20, 30, 40, 50, 60, 70],
    #    'knn__metric': ['cosine', 'euclidean', 'minkowski'], 
    # }

    param_distributions = {
        'customcebra__temperature': (0.11, 3.21),
        'customcebra__time_offsets': (1, 10),
        'customcebra__output_dimension': (3, 15),
        'customcebra__batch_size': [256, 512, 1024],
        'customcebra__learning_rate': (1e-5, 1e-2),
        'knn__n_neighbors': (2, 70),
        'knn__metric': ['cosine', 'euclidean', 'minkowski'], 
    }


    # get current date and time
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y_%H-%M-%S") 

    # will use k-folds with 10 splits
    n_splits = 5 # 10
    n_timesteps = inputs.shape[0]
    num_windows = 340 # 1000
    folds = create_folds(n_timesteps, num_folds=n_splits, num_windows=num_windows)
    folds_file_name = f'custom_folds_{animal}_{session}_{date_time}'
    folds_file_path = os.path.join(model_dir, folds_file_name + '.pkl')
    with open(folds_file_path, 'wb') as f:
        pickle.dump(folds, f)

    # set up logging
    log_file = os.path.join(model_dir, f'grid_search_{animal}_{session}_{date_time}.log')
    # logging.basicConfig(filename=f'grid_search_{date_time}.log', level=logging.DEBUG)
    
    # Define the grid search
    logger = Logger(log_file)

    scorer = make_scorer(logger.log_and_score)
    # clf = RandomizedSearchCV(pipe, param_distributions, n_iter=200, cv=folds, scoring=scorer, n_jobs=-1, random_state=0, verbose=3)
    clf = BayesSearchCV(pipe, param_distributions, n_iter=200, cv=folds, scoring=scorer, n_jobs=-1, random_state=0, verbose=3)
    search = clf.fit(inputs, labels)
    
    # save the search
    search_file_name = f'grid_search_model_{animal}_{session}_{date_time}'
    search_file_path = os.path.join(model_dir, search_file_name + '.pkl')
    with open(search_file_path, 'wb') as f:
        pickle.dump(search, f)

    # print "saved_model"
    print("saved_model")

    # print best parameters
    print(search.best_params_)
    print(search.best_score_)

    

if __name__ == "__main__":
    main()
    

            
        
