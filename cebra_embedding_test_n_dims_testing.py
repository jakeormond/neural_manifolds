import sys
# import logging
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
# from matplotlib.collections import LineCollection
import os
# import cebra.datasets
import logging
import cebra
import torch
from cebra import CEBRA
from sklearn.model_selection import KFold, ParameterGrid, PredefinedSplit, ParameterSampler, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.base import BaseEstimator
import scipy.ndimage
from sklearn.metrics import make_scorer, r2_score
# from skopt import BayesSearchCV
import pickle

from cebra_embedding import create_folds
from cebra_embedding_time_grid_search import CustomCEBRA, Logger 

# sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
# from utilities.get_directories import get_data_dir


def main():    
        
    # check if pytorch has access to GPU
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    animal = 'rat_9'
    session = '10-12-2021'

    # create model dir if it doesn't exits
    model_dir = os.path.join('D:/analysis/og_honeycomb/models', animal, 'n_dimensions')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # load best model from grid search
    # model_dir = os.path.join('/ceph/scratch/jakeo/honeycomb_neural_data/models/', animal)
    grid_dir = os.path.join('D:/analysis/og_honeycomb/models', animal, 'z_scored_spike_xy_goaldir')
    model_files = [f for f in os.listdir(grid_dir) if f.startswith('grid_search_model_')]
    model_file = model_files[0]
    model_path = os.path.join(grid_dir, model_file)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)        
    best_params = model.best_params_
    print(best_params)
    # delete model var
    del model

    # load the folds from the grid_dir
    folds_files = [f for f in os.listdir(grid_dir) if f.startswith('custom_folds')]
    folds_file = folds_files[0]
    folds_path = os.path.join(grid_dir, folds_file)
    with open(folds_path, 'rb') as f:
        folds = pickle.load(f)

    # load the data
    # data_dir = os.path.join('/ceph/scratch/jakeo/honeycomb_neural_data/', animal, session)
    data_dir = os.path.join('D:/analysis/carlas_windowed_data/honeycomb_neural_data', animal, session)
    
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

    # Define the parameter grid
    param_grid = best_params
    # individual values in param grid need to be wrapped in lists
    for key in param_grid.keys():
        param_grid[key] = [param_grid[key]]    

    param_grid['customcebra__output_dimension'] = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]

    # get current date and time
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y_%H-%M-%S") 

      # set up logging
    log_file = os.path.join(model_dir, f'grid_search_{animal}_{session}_{date_time}.log')
    logging.basicConfig(filename=f'grid_search_{date_time}.log', level=logging.DEBUG)
    
    # Define the grid search
    logger = Logger(log_file)

    scorer = make_scorer(logger.log_and_score)
    # clf = RandomizedSearchCV(pipe, param_distributions, n_iter=200, cv=folds, scoring=scorer, n_jobs=-1, random_state=0, verbose=3)
    # clf = BayesSearchCV(pipe, param_distributions, n_iter=200, cv=folds, scoring=scorer, n_jobs=-1, random_state=0, verbose=3)
    clf = GridSearchCV(pipe, param_grid, cv=folds, scoring=scorer, n_jobs=-1, verbose=3)
    search = clf.fit(inputs, labels)
    # print all the parameters
    print(search.cv_results_)
    
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
    

            
        
