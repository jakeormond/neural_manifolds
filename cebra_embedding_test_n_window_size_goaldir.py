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
from sklearn.model_selection import KFold, ParameterGrid, PredefinedSplit, ParameterSampler, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.base import BaseEstimator
import scipy.ndimage
from sklearn.metrics import make_scorer, r2_score
# from skopt import BayesSearchCV
import pickle
from datetime import datetime

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
    model_dir = os.path.join('D:/analysis/og_honeycomb/models', animal, 'window_size_for_goaldir')
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

    # data_dir = os.path.join('/ceph/scratch/jakeo/honeycomb_neural_data/', animal, session)
    data_dir = os.path.join('D:/analysis/carlas_windowed_data/honeycomb_neural_data', animal, session)

    # Define the CEBRA model
    cebra_model = CustomCEBRA()

    # Define the knn regressor
    knn = KNeighborsRegressor()
    
    window_sizes = [20, 50, 100, 250, 500]

    # create a data frame to store the scores with the window sizes as columns, and each of the 5 splits as rows
    scores_df = pd.DataFrame(columns=window_sizes)

    for w in window_sizes:

        # load the data
        dlc_dir = os.path.join(data_dir, 'positional_data')
        labels = np.load(f'{dlc_dir}/labels_{w}.npy')
        # keep the first 2 columns and the last 2 columns
        # labels = labels[:, [0, 1, -2, -1]]
        # just direction to goal
        labels = labels[:, [-2, -1]]

        spike_dir = os.path.join(data_dir, 'physiology_data')
        spike_data = np.load(f'{spike_dir}/inputs_zscored_{w}.npy')

        # convert inputs to torch tensor
        inputs = torch.tensor(spike_data, dtype=torch.float32)  

        # create the folds
        n_splits = 5 # 10
        n_timesteps = inputs.shape[0]
        num_windows = 340 # 1000
        folds = create_folds(n_timesteps, num_folds=n_splits, num_windows=num_windows)

        now = datetime.now()
        date_time = now.strftime("%d-%m-%Y_%H-%M-%S") 

        folds_file_name = f'custom_folds_{animal}_{session}_{date_time}_window_size_{w}'
        folds_file_path = os.path.join(model_dir, folds_file_name + '.pkl')
        with open(folds_file_path, 'wb') as f:
            pickle.dump(folds, f)

        # Define the pipeline
        pipe = Pipeline(steps=[('customcebra', cebra_model), ('knn', knn)])
        pipe.set_params(**best_params)

        # set the customcebra output dimension to 10
        pipe.set_params(customcebra__output_dimension=10)

        # set the customcebra max_iterations to 100
        pipe.set_params(customcebra__max_iterations=100)

        

        # set up logging
        log_file = os.path.join(model_dir, f'goaldir_{animal}_{session}_{date_time}_window_size_{w}.log')
        # logging.basicConfig(filename=f'grid_search_{date_time}.log', level=logging.DEBUG)
        
        # Define the grid search
        logger = Logger(log_file)

        scorer = make_scorer(logger.log_and_score)
        scores = cross_val_score(pipe, inputs, labels, cv=folds, scoring=scorer, n_jobs=-1)
        scores_df[w] = scores
       
    scores_df.to_csv(os.path.join(model_dir, 'r2_scores.csv'))



if __name__ == "__main__":
    main()
    

            
        
