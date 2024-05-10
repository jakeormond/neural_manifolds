import os
import pickle
from sklearn.metrics import r2_score
 
from cebra_embedding_time_grid_search import CustomCEBRA, Logger # needed because I saved the model as a pickle file





def main():   

    animal = 'rat_7'
    session = '6-12-2019'

    model_dir = os.path.join('D:/analysis/og_honeycomb', animal, session, 'cebra_grid_search') 

    # find the model pickle file in model_dir; it begins "cebra_grid_search_"
    model_files = [f for f in os.listdir(model_dir) if f.startswith('grid_search_model_')]
    if len(model_files) != 1:
        raise ValueError(f"Expected 1 model file in {model_dir}, found {len(model_files)}")
    model_file = model_files[0]

    # load the model pickle file
    model_path = os.path.join(model_dir, model_file)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # find the best parameters
    best_params = model.best_params_
    print(best_params)

    # find the best score
    best_score = model.best_score_
    print(best_score)

    # make a list of all the model parameters evaluated during the grid search
    all_params = model.cv_results_['params']

    ########### make plots of the grid search results ############
    params = list(best_params.keys())

    for p in params:

        # make a list of the unique values of the parameter p
        values = list(set([d[p] for d in all_params]))

        # make a list of the mean test scores for each value of the parameter p
        scores = [d['mean_test_score'] for d in model.cv_results_ if d[p] == values[0]]

        # make a plot of the mean test scores for each value of the parameter p
        plt.plot(values, scores)
        plt.xlabel(p)
        plt.ylabel('mean test score')
        plt.title(f'mean test score vs {p}')
        plt.show()



    # find the best estimator
    best_estimator = model.best_estimator_

    





    pass 




if __name__ == "__main__":
    main()



