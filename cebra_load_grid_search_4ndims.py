import os
import pickle
import pandas as pd
from sklearn.metrics import r2_score
import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
 
from cebra_embedding_time_grid_search import CustomCEBRA, Logger # needed because I saved the model as a pickle file


def plot_grid_out_dimension_vs_r2(model, grid_dir):
    ########### PLOT THE RESULTS FROM THE LOADED GRID SEARCH ############
    # plot the r2 scores agains the number of dimensions
    plt.figure()
    results_df = pd.DataFrame(model.cv_results_)
    plt.plot(results_df['param_customcebra__output_dimension'], results_df['mean_test_score'])
    plt.xlabel('Number of dimensions')
    plt.ylabel('Mean test score')
    plt.title('Number of dimensions vs mean test score')
    plt.savefig(os.path.join(grid_dir, 'n_dimensions_vs_mean_test_score.png'))


def individual_r2_scores(folds, inputs, labels, best_params):
    # Define the CEBRA model
    cebra_model = CustomCEBRA()

    # Define the knn regressor
    knn = KNeighborsRegressor()

    # Define the pipeline
    pipe = Pipeline(steps=[('customcebra', cebra_model), ('knn', knn)])

    # train a model for each fold using the pipeline with the best params
    r2_scores = {'all': [], 'x': [], 'y': [], 'goal_sin': [], 'goal_cos': []}
    models = []
    for i, (train_index, test_index) in enumerate(folds):
        print(f'Fold {i+1} of {len(folds)}')
        X_train, X_test = inputs[train_index,:], inputs[test_index,:]
        y_train, y_test = labels[train_index,:], labels[test_index,:]

        # set the pipeline parameters
        pipe.set_params(**best_params)

        # Train the model on the training data
        pipe.fit(X_train, y_train)

        models.append(pipe)

        # Predict the test data
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        r2_scores['all'].append(r2)

        # r2 for x
        r2 = r2_score(y_test[:, 0], y_pred[:, 0])
        r2_scores['x'].append(r2)

        # r2 for y
        r2 = r2_score(y_test[:, 1], y_pred[:, 1])
        r2_scores['y'].append(r2)

        # r2 for goal_sin
        r2 = r2_score(y_test[:, 2], y_pred[:, 2])
        r2_scores['goal_sin'].append(r2)

        # r2 for goal_cos
        r2 = r2_score(y_test[:, 3], y_pred[:, 3])
        r2_scores['goal_cos'].append(r2)

    
    # save the r2 scores
    r2_scores_df = pd.DataFrame(r2_scores)
    # make a final row with the mean of the r2 scores
    r2_scores_df.loc['mean'] = r2_scores_df.mean()

    return r2_scores_df, models

def main():   

    # animals = ['rat_3', 'rat_7', 'rat_8', 'rat_9', 'rat_10']
    animals = ['rat_7', 'rat_9']
    sessions = {'rat_3': '25-3-2019', 'rat_7': '6-12-2019', 'rat_8': '15-10-2019', 'rat_9': '10-12-2021', 'rat_10': '23-11-2021'}

    model_dir = os.path.join('D:/analysis/og_honeycomb/models') 

    for a in animals:
        animal_dir = os.path.join(model_dir, a)
        session = sessions[a]

        grid_dir = os.path.join(animal_dir, 'n_dimensions')

        # if grid_dir doesn't exist, continue
        if not os.path.exists(grid_dir):
            print(f"Grid search directory {grid_dir} doesn't exist")
            continue
        
        # find the model pickle file in model_dir; it begins "cebra_grid_search_"
        model_files = [f for f in os.listdir(grid_dir) if f.startswith('grid_search_model_')]
        if len(model_files) == 0:
            print(f"No model files found in {model_dir}")
            continue
        
        # elif len(model_files) != 1:
        #     raise ValueError(f"Expected 1 model file in {model_dir}, found {len(model_files)}")
        
        model_file = model_files[0]

        # load the model pickle file
        model_path = os.path.join(grid_dir, model_file)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)   
        # get best params
        best_params = model.best_params_

        # plot the results
        plot_grid_out_dimension_vs_r2(model, grid_dir)
       
        # load the cv folds
        folds_dir = os.path.join('D:/analysis/og_honeycomb/models', a, 'z_scored_spike_xy_goaldir')
        folds_files = [f for f in os.listdir(folds_dir) if f.startswith('custom_folds')]
        folds_file = folds_files[0]
        folds_path = os.path.join(folds_dir, folds_file)
        with open(folds_path, 'rb') as f:
            folds = pickle.load(f)

        # load the data
        # data_dir = os.path.join('/ceph/scratch/jakeo/honeycomb_neural_data/', animal, session)
        data_dir = os.path.join('D:/analysis/carlas_windowed_data/honeycomb_neural_data', a, session)
        
        dlc_dir = os.path.join(data_dir, 'positional_data')
        labels = np.load(f'{dlc_dir}/labels_250.npy')
        # keep the first 2 columns and the last 2 columns
        labels = labels[:, [0, 1, -2, -1]]

        spike_dir = os.path.join(data_dir, 'physiology_data')
        spike_data = np.load(f'{spike_dir}/inputs_zscored_250.npy')

        # convert inputs to torch tensor
        inputs = torch.tensor(spike_data, dtype=torch.float32)  

        # train a model for each fold using the pipeline with the best params
        r2_scores_df, models = individual_r2_scores(folds, inputs, labels, best_params)

        r2_scores_df.to_csv(os.path.join(grid_dir, 'r2_scores.csv'))

        # save the models
        for i, model in enumerate(models):
            model_file = os.path.join(grid_dir, f'model_{i}.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        


     
    pass


            
            




if __name__ == "__main__":
    main()

    pass



