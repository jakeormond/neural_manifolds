import torch
import sys
import os
import pickle
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor
import cebra
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
# sys.path.append('/home/jake/Documents/python_code/robot_maze_analysis_code')
from utilities.get_directories import get_data_dir
from utilities.load_and_save_data import save_pickle, load_pickle

from cebra_embedding import create_folds


def create_folds_v2(n_timesteps, num_folds=5, num_windows=10):
    n_windows_total = num_folds * num_windows
    window_size = n_timesteps // n_windows_total
    window_start_ind = np.arange(0, n_timesteps, window_size)

    folds = []

    for i in range(num_folds):
        # Uniformly select test windows from the total windows
        step_size = n_windows_total // num_windows
        test_windows = np.arange(i, n_windows_total, step_size)
        test_ind = []
        for j in test_windows:
            # Select every nth index for testing, where n is the step size
            test_ind.extend(np.arange(window_start_ind[j], window_start_ind[j] + window_size, step_size))
        train_ind = list(set(range(n_timesteps)) - set(test_ind))

        folds.append((train_ind, test_ind))

    # As a sanity check, plot the distribution of the test indices
    # fig, ax = plt.subplots()
    # ax.hist(train_ind, label='train')
    # ax.hist(test_ind, label='test')
    # ax.legend()
    # plt.show()

    return folds



def decode_pos(emb_train, emb_test, label_train, n_neighbors=36):
    pos_decoder = KNeighborsRegressor(n_neighbors)

    pos_decoder.fit(emb_train, label_train)
    pos_pred = pos_decoder.predict(emb_test)

    return pos_pred

def load_cebra_model(model_name, data_dir, goal, window_size, i):
    model_name = f'cebra_{model_name}_goal{goal}_ws{window_size}_fold{i+1}.pt'
    model_path = os.path.join(data_dir, model_name)
    cebra_pos3_model = cebra.CEBRA.load(model_path)

    return cebra_pos3_model, model_name


def get_embeddings(model, inputs, train_index, test_index):
    emb_train = model.transform(inputs[train_index, :])
    emb_test = model.transform(inputs[test_index, :])

    return emb_train, emb_test


def colormap_2d():
    # get the viridis colormap
    v_cmap = plt.get_cmap('viridis')
    v_colormap_values = v_cmap(np.linspace(0, 1, 256))

    # get the cool colormap
    c_cmap = plt.get_cmap('cool')
    c_colormap_values = c_cmap(np.linspace(0, 1, 256))

    # get the indices of each colormap for the 2d map
    v_v, c_v = np.meshgrid(np.arange(256), np.arange(256))

    # create a new 2d array with the values of the colormap
    colormap = np.zeros((256, 256, 4))

    for x in range(256):
        for y in range(256):
            v_val = v_colormap_values[v_v[x, y], :]
            c_val = c_colormap_values[c_v[x, y], :]

            # take the average of the two colormaps
            colormap[x, y, :] = (v_val + c_val) / 2

    return colormap


def get_color_from_position(position, colormap):
    x = position[:,0]
    y = position[:,1]

    # convert x and y into vectors of integers between 0 and 255
    x_int = np.interp(x, (x.min(), x.max()), (0, 255)).astype(int)
    y_int = np.interp(y, (y.min(), y.max()), (0, 255)).astype(int)
    color_data = colormap[x_int, y_int]

    return color_data


if __name__ == "__main__":
    animal = 'Rat46'
    session = '19-02-2024'
    data_dir = get_data_dir(animal, session)

    goal = 52
    # window_size = 100
    

    # file_name = f'decoding_scores_goal{goal}_ws{window_size}.pkl'
    # file_path = os.path.join(data_dir, file_name)

    # with open(file_path, 'rb') as f:
    #     decoding_scores = pickle.load(f)


    ########## CREATE LIST OF MODELS ###############
    # model_list = ['time3', 'pos3', 'pos_hybrid3', 'pos_shuffled3']
    model_list = ['time3', 'time_shuffled3']
    # model_list = ['pos3']

    ########## CREATE LIST OF TIMEWINDOWS #############
    # window_sizes = [25, 50, 100, 250, 500]
    # window_sizes = [100, 250, 500]
    window_sizes = [100, 250]


    ########## CREATE FIGURE OF EMBEDDINGS AND DECODING #######################

    # folds_by_model_and_window = {}

    for m in model_list:
        for window_size in window_sizes:
         
            ############# LOAD FOLDS #################        
            folds_file = f'folds_goal{goal}_ws{window_size}.pkl'
            # load the file
            folds = load_pickle(folds_file, data_dir)
            n_folds = len(folds)

            ############ LOAD POSITIONAL DATA ################
            dlc_dir = os.path.join(data_dir, 'deeplabcut', 'labels_for_embedding_and_decoding')
            labels_file_name = f'labels_goal{goal}_ws{window_size}'
            # load numpy array of labels
            labels = np.load(os.path.join(dlc_dir, labels_file_name + '.npy'))
            # keep only the first 2 columns
            labels = labels[:, :2]

            ############ LOAD SPIKE DATA #####################
            # load numpy array of neural data
            spike_dir = os.path.join(data_dir, 'spike_sorting', 'inputs_for_embedding_and_decoding')
            inputs_file_name = f'inputs_goal{goal}_ws{window_size}'
            inputs = np.load(os.path.join(spike_dir, inputs_file_name + '.npy'))

            # load convert inputs to torch tensor
            inputs = torch.tensor(inputs, dtype=torch.float32)  

            # will use k-folds with 5 splits
            # n_splits = 5
            # kf = KFold(n_splits=n_splits, shuffle=False)
            # n_timesteps = inputs.shape[0]
            # num_windows = 10
            # folds = create_folds(n_timesteps, num_folds=n_splits, num_windows=num_windows)
            # folds_v2 = create_folds_v2(n_timesteps, num_folds=5, num_windows=10)

            # dict_key = f'{m}_ws{window_size}'

            test_embeddings = []
            test_labels = []

            # create figure with 3 rows and 6 columns
            fig = plt.figure(figsize=(24, 24), dpi=100)
            colormap = colormap_2d()  
            

            for i, (train_index, test_index) in enumerate(folds):
               

                model, model_name = load_cebra_model(m, data_dir, goal, window_size, i)
                
                emb_train, emb_test = get_embeddings(model, inputs, train_index, test_index)

                # convert positional data to colors
                colordata = get_color_from_position(labels, colormap)
                colordata_train = colordata[train_index, :]
                colordata_test = colordata[test_index, :]

                # create subplot of 3d embeddings, with projection='3d'
                ax = fig.add_subplot(6, n_folds+1, i+1, projection='3d')
                # ax.scatter(emb_train[::10, 0], emb_train[::10, 1], emb_train[::10,2], c=colordata_train[::10,:], s=0.5)
                ax.scatter(emb_train[:, 0], emb_train[:, 1], emb_train[:,2], c=colordata_train, s=0.5)

                ax = fig.add_subplot(6, n_folds+1, (n_folds+1)*3 + i+1, projection='3d')
                # ax.scatter(emb_test[::10, 0], emb_test[::10, 1], emb_test[::10,2], c=colordata_test[::10,:], s=0.5)
                ax.scatter(emb_test[:, 0], emb_test[:, 1], emb_test[:, 2], c=colordata_test, s=0.5)
                
                # decoding
                label_train = labels[train_index, :]
                label_test = labels[test_index, :]

                decoded_train_pos = decode_pos(emb_train, emb_train, label_train, n_neighbors=72)
                decoded_test_pos = decode_pos(emb_train, emb_test, label_train, n_neighbors=72)  
    
                for j in range(2):
                    ax = fig.add_subplot(6, n_folds+1, (n_folds+1)*(j+1) + i+1)
                    ax.scatter(label_train[:, j], decoded_train_pos[:, j], 0.5, 'b')
                    r2_val = sklearn.metrics.r2_score(label_train[:, j], decoded_train_pos[:, j])
                    # plot r2_val on plot
                    ax.text(0.5, 0.5, f'R2: {r2_val:.2f}', fontsize=16)

                    ax = fig.add_subplot(6, n_folds+1, (n_folds+1)*(j+4) + i+1)
                    ax.scatter(label_test[:, j], decoded_test_pos[:, j], 0.5, 'r')
                    r2_val = sklearn.metrics.r2_score(label_test[:, j], decoded_test_pos[:, j])
                    # plot r2_val on plot
                    ax.text(0.5, 0.5, f'R2: {r2_val:.2f}', fontsize=16)
        
            # plot the colormap in the sixth column of the first row
            ax = fig.add_subplot(6, n_folds+1, n_folds+1)
            ax.imshow(colormap)
            
            fig_name = f'{model_name}_embedding_and_decoding_goal{goal}_ws{window_size}.png'
            fig_path = os.path.join(data_dir, fig_name)
            fig.savefig(fig_path)
            pass

    # save the folds_by_model_and_window dictionary
    # folds_by_model_and_window_name = f'folds_by_model_and_window_goal{goal}.pkl'
    # folds_by_model_and_window_path = os.path.join(data_dir, folds_by_model_and_window_name)
    # save_pickle(folds_by_model_and_window, folds_by_model_and_window_name, data_dir)


    ######################################### PLOT THE LOSS #########################################
    fig = plt.figure(figsize=(25,4))
    colours = ['blue', 'orange', 'green', 'red']
    for i in range(n_splits):
        ax = fig.add_subplot(1, 5, i+1)
        for j, m in enumerate(model_list):  
            model, model_name = load_cebra_model(m, data_dir, goal, window_size, i)        
            ax.plot(model.state_dict_['loss'], c=colours[j], label = m)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')   

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('InfoNCE Loss')
        plt.legend(bbox_to_anchor=(0.5,0.3), frameon = False )
        # plt.show()

    fig_name = f'loss_goal{goal}_ws{window_size}.png'
    fig_path = os.path.join(data_dir, fig_name)
    fig.savefig(fig_path)