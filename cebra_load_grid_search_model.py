import os
import pickle
import pandas as pd
from sklearn.metrics import r2_score
 
from cebra_embedding_time_grid_search import CustomCEBRA, Logger # needed because I saved the model as a pickle file





def main():   

    animals = ['rat_3', 'rat_7', 'rat_8', 'rat_9', 'rat_10']

    model_dir = os.path.join('D:/analysis/og_honeycomb/models') 

    counter = 0
    for a in animals:
        animal_dir = os.path.join(model_dir, a)

        for z in range(3):
            if z == 0:
                grid_dir = os.path.join(animal_dir, 'non_z_scored_spike_trains')
                title = f'{a}_non_z_scored_spike_trains'
            elif z == 1:
                grid_dir = os.path.join(animal_dir, 'z_scored_spike_trains')
                title = f'{a}_z_scored_spike_trains'
            else:
                grid_dir = os.path.join(animal_dir, 'z_scored_spike_xy_goaldir')
                title = f'{a}_z_scored_spike_xy_goaldir'
            
            # check if grid_dir exists
            if not os.path.exists(grid_dir):
                print(f"{grid_dir} does not exist")
                continue

            # find the model pickle file in model_dir; it begins "cebra_grid_search_"
            model_files = [f for f in os.listdir(grid_dir) if f.startswith('grid_search_model_')]
            if len(model_files) == 0:
                print(f"No model files found in {model_dir}")
                continue
            
            elif len(model_files) != 1:
                raise ValueError(f"Expected 1 model file in {model_dir}, found {len(model_files)}")
                
            model_file = model_files[0]

            # load the model pickle file
            model_path = os.path.join(grid_dir, model_file)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)            

            # get the results
            # results_df = pd.DataFrame(model.cv_results_)
            # # order the results by rank_test_score
            # results_df.sort_values(by='rank_test_score', inplace=True)
            # # save the results to a csv file
            # results_df.to_csv(os.path.join(grid_dir, 'grid_search_results.csv'), index=False)


            # find the best parameters
            best_params = model.best_params_
            print(best_params)

            # find the best score
            best_score = model.best_score_
            print(best_score)

            ##### make dataframe of the best_params #####
            if counter == 0:
                grid_results = pd.DataFrame(best_params, index=[title])
                # add best_score to the dataframe
                grid_results['best_score'] = best_score
            else:
                temp_df = pd.DataFrame(best_params, index=[title])
                # add best_score to the dataframe
                temp_df['best_score'] = best_score
                grid_results = pd.concat([grid_results, temp_df]) 
            
            counter += 1
    
    
    print(grid_results)
    pass


            
            




if __name__ == "__main__":
    main()

    pass



