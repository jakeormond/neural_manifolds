import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.collections import LineCollection
import os
import cebra.datasets
import cebra
import torch
from cebra import CEBRA
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
import pickle

# sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
# from utilities.get_directories import get_data_dir


def create_folds(n_timesteps, num_folds=5, num_windows=4):
    n_windows_total = num_folds * num_windows
    window_size = n_timesteps // n_windows_total
    window_start_ind = np.arange(0, n_timesteps, window_size)

    folds = []

    for i in range(num_folds):
        test_windows = np.arange(i, n_windows_total, num_folds)
        test_ind = []
        for j in test_windows:
            test_ind.extend(np.arange(window_start_ind[j], window_start_ind[j] + window_size))
        train_ind = list(set(range(n_timesteps)) - set(test_ind))

        folds.append((train_ind, test_ind))

    return folds


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
    
    pos_test_x_score = sklearn.metrics.r2_score(label_test[:, 0], pos_pred[:,0])
    pos_test_y_score = sklearn.metrics.r2_score(label_test[:, 1], pos_pred[:,1])

    return pos_test_err, pos_test_x_score, pos_test_y_score


def plot_embeddings(embeddings):
    pass


''' using data created with pytorch_decoding.dataset_creation.py '''

if __name__ == "__main__":
    animal = 'Rat46'
    session = '19-02-2024'
    # data_dir = get_data_dir(animal, session)

    data_dir = '/ceph/scratch/jakeo/'

    goal = 52
    window_size = 100


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
    n_splits = 5
    # kf = KFold(n_splits=n_splits, shuffle=False)
    n_timesteps = inputs.shape[0]
    folds = create_folds(n_timesteps, num_folds=n_splits, num_windows=4)
    
    cebra_pos3_embeddings = []
    cebra_pos3_labels = []
    cebra_pos3_decoding_score = []

    cebra_pos_shuffled3_embeddings = []
    cebra_pos_shuffled3_labels = []
    cebra_pos_shuffled3_decoding_score = []

    cebra_time3_embeddings = []
    cebra_time3_labels = []
    cebra_time3_decoding_score = []

    cebra_pos_hybrid3_embeddings = []
    cebra_pos_hybrid3_labels = []    
    cebra_pos_hybrid3_decoding_score = []

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
    
        cebra_file_name = f'cebra_pos3_model_goal{goal}_ws{window_size}_fold{i+1}.pt'
        cebra_file_path = os.path.join(cebra_model_dir, cebra_file_name)
        # need to convert any double backslashes to forward slashes      
        cebra_file_path = cebra_file_path.replace("\\", "/")        
        cebra_pos3_model.save(cebra_file_path)

        ########## GET THE TRAIN AND TEST EMBEDDINGS ##################
        cebra_pos3_train_embeddings = cebra_pos3_model.transform(X_train)
        cebra_pos3_test_embeddings = cebra_pos3_model.transform(X_test)

        cebra_pos3_embeddings.append(cebra_pos3_test_embeddings)
        cebra_pos3_labels.append(y_test)

        ####### CALCULATE THE DECODING SCORES ########
        cebra_pos3_decoding_score.append(decoding_pos(cebra_pos3_train_embeddings, cebra_pos3_test_embeddings, y_train, y_test, n_neighbors=36))

        ################ SHUFFLED BEHAVIOUR MODEL ##################
        cebra_pos_shuffled3_model = CEBRA(model_architecture='offset10-model',
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

        X_train_shuffled = np.random.permutation(X_train)

        cebra_pos_shuffled3_model.fit(X_train_shuffled, y_train)
        print('finished fitting shuffle model')
    
        cebra_file_name = f'cebra_pos_shuffled3_model_goal{goal}_ws{window_size}_fold{i+1}.pt'
        cebra_file_path = os.path.join(cebra_model_dir, cebra_file_name)
        # need to convert any double backslashes to forward slashes      
        cebra_file_path = cebra_file_path.replace("\\", "/")        
        cebra_pos_shuffled3_model.save(cebra_file_path)

        ########## GET THE TEST EMBEDDINGS ##################
        cebra_pos_shuffled3_train_embeddings = cebra_pos_shuffled3_model.transform(X_train)
        cebra_pos_shuffled3_test_embeddings = cebra_pos_shuffled3_model.transform(X_test)        
        
        cebra_pos_shuffled3_embeddings.append(cebra_pos_shuffled3_test_embeddings)
        cebra_pos_shuffled3_labels.append(y_test)

        ####### CALCULATE THE DECODING SCORES ########
        cebra_pos_shuffled3_decoding_score.append(decoding_pos(cebra_pos_shuffled3_train_embeddings, cebra_pos_shuffled3_test_embeddings, y_train, y_test, n_neighbors=36))


        ################## TIME-ONLY MODEL ########################
        cebra_time3_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1.12,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

        cebra_time3_model.fit(X_train)
        print('finished fitting time model')

        cebra_file_name = f'cebra_time3_model_goal{goal}_ws{window_size}_fold{i+1}.pt'
        cebra_file_path = os.path.join(cebra_model_dir, cebra_file_name)
        # need to convert any double backslashes to forward slashes      
        cebra_file_path = cebra_file_path.replace("\\", "/")        
        cebra_time3_model.save(cebra_file_path)

        ########## GET THE TEST EMBEDDINGS ##################
        cebra_time3_train_embeddings = cebra_time3_model.transform(X_train)
        cebra_time3_test_embeddings = cebra_time3_model.transform(X_test)

        cebra_time3_embeddings.append(cebra_time3_test_embeddings)
        cebra_time3_labels.append(y_test)

        ####### CALCULATE THE DECODING SCORES ########
        cebra_time3_decoding_score.append(decoding_pos(cebra_time3_train_embeddings, cebra_time3_test_embeddings, y_train, y_test, n_neighbors=36))


        ################## BEHAVIOUR MODEL WITH POSITION - HYBRID ########################
        cebra_pos_hybrid3_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10,
                        hybrid=True)

        cebra_pos_hybrid3_model.fit(X_train, y_train)
        print('finished fitting hybrid model')

        cebra_file_name = f'cebra_pos_hybrid3_model_goal{goal}_ws{window_size}_fold{i+1}.pt'
        cebra_file_path = os.path.join(cebra_model_dir, cebra_file_name)
        # need to convert any double backslashes to forward slashes
        cebra_file_path = cebra_file_path.replace("\\", "/")
        cebra_pos_hybrid3_model.save(cebra_file_path)

        ########### GET THE TEST EMBEDDINGS ##################
        cebra_pos_hybrid3_train_embeddings = cebra_pos_hybrid3_model.transform(X_train)
        cebra_pos_hybrid3_test_embeddings = cebra_pos_hybrid3_model.transform(X_test)

        cebra_pos_hybrid3_embeddings.append(cebra_pos_hybrid3_test_embeddings)
        cebra_pos_hybrid3_labels.append(y_test)

        ####### CALCULATE THE DECODING SCORES ########
        cebra_pos_hybrid3_decoding_score.append(decoding_pos(cebra_pos_hybrid3_train_embeddings, cebra_pos_hybrid3_test_embeddings, y_train, y_test, n_neighbors=36))


    ############ SAVE THE DECODING SCORES ################
    decoding_scores = {'cebra_pos3_decoding_score': cebra_pos3_decoding_score,
                          'cebra_pos_shuffled3_decoding_score': cebra_pos_shuffled3_decoding_score,
                          'cebra_time3_decoding_score': cebra_time3_decoding_score,
                          'cebra_pos_hybrid3_decoding_score': cebra_pos_hybrid3_decoding_score} 
    decoding_score_file_path = os.path.join(cebra_model_dir, f'decoding_scores_goal{goal}_ws{window_size}.pkl')
    with open(decoding_score_file_path, 'wb') as f:
        pickle.dump(decoding_scores, f)
    
    ############## PLOT THE EMBEDDING ################
    # create a figure
    fig = plt.figure(figsize = (24,18), dpi = 100)
    colormap = colormap_2d()  

    # loop through the test_embeddings list
    for i, embedding in enumerate(cebra_pos3_embeddings):
        #### PLOT POSITION EMBEDDING ####
        # create a new subplot
        ax = fig.add_subplot(4, n_splits+1, i+1, projection='3d')
        # plot the embedding
        embedding_data = cebra_pos3_embeddings[i][::10,:]

        # positional data
        pos_data = test_labels[i][::10,:]
        x = pos_data[:,0]
        y = pos_data[:,1]

        # convert x and y into vectors of integers between 0 and 255
        x_int = np.interp(x, (x.min(), x.max()), (0, 255)).astype(int)
        y_int = np.interp(y, (y.min(), y.max()), (0, 255)).astype(int)
        color_data = colormap[x_int, y_int]
        
        ax.scatter(embedding_data[:,0], embedding_data[:,1], embedding_data[:,2], c=color_data, s=1)

        ##### PLOT SHUFFLED EMBEDDING ####
        # create a new subplot
        ax = fig.add_subplot(4, n_splits+1, i+1+n_splits+1, projection='3d')
        # plot the embedding
        embedding_data = cebra_pos_shuffled3_embeddings[i][::10,:]
        ax.scatter(embedding_data[:,0], embedding_data[:,1], embedding_data[:,2], c=color_data, s=1)

        ##### PLOT TIME EMBEDDING ####
        # create a new subplot
        ax = fig.add_subplot(4, n_splits+1, i+1+2*n_splits+1, projection='3d')
        # plot the embedding
        embedding_data = cebra_time3_embeddings[i][::10,:]
        ax.scatter(embedding_data[:,0], embedding_data[:,1], embedding_data[:,2], c=color_data, s=1)

        ##### PLOT HYBRID EMBEDDING ####
        # create a new subplot
        ax = fig.add_subplot(4, n_splits+1, i+1+3*n_splits+1, projection='3d')
        # plot the embedding
        embedding_data = cebra_pos_hybrid3_embeddings[i][::10,:]
        ax.scatter(embedding_data[:,0], embedding_data[:,1], embedding_data[:,2], c=color_data, s=1)
    
    # plot the colormap
    ax = fig.add_subplot(4, n_splits+1, n_splits+1)
    ax.imshow(colormap)

    # fig_name = cebra_file_name.replace(".pt", ".png")
    fig_name = f"cebra_pos_shuffled3_time_hybrid_goal{goal}_ws{window_size}.png"
    
    # save fig to the cebra directory
    fig_dir = data_dir
    fig_path = os.path.join(fig_dir, fig_name)
    plt.savefig(fig_path)

    ################ PLOT THE LOSS #############################
    fig = plt.figure(figsize=(25,4))

    for i in range(n_splits):
        ax = plt.subplot(1,n_splits, i+1)
        
        # load models
        cebra_file_name = f'cebra_pos3_model_goal{goal}_ws{window_size}_fold{i+1}.pt'
        cebra_file_path = os.path.join(cebra_model_dir, cebra_file_name)
        cebra_file_path = cebra_file_path.replace("/", "\\")        
        cebra_pos3_model = cebra.CEBRA.load(cebra_file_path)
        ax.plot(cebra_pos3_model.state_dict_['loss'], c='deepskyblue', label = 'position')
        
        cebra_file_name = f'cebra_pos_shuffled3_model{goal}_ws{window_size}_fold{i+1}.pt'
        cebra_file_path = os.path.join(cebra_model_dir, cebra_file_name)
        cebra_file_path = cebra_file_path.replace("/", "\\")        
        cebra_pos_shuffled3_model = cebra.CEBRA.load(cebra_file_path)
        ax.plot(cebra_pos_shuffled3_model.state_dict_['loss'], c='gray', label = 'shuffled')
        
        cebra_file_name = f'cebra_time3_model_goal{goal}_ws{window_size}_fold{i+1}.pt'
        cebra_file_path = os.path.join(cebra_model_dir, cebra_file_name)
        cebra_file_path = cebra_file_path.replace("/", "\\")        
        cebra_time3_model = cebra.CEBRA.load(cebra_file_path)       
        ax.plot(cebra_time3_model.state_dict_['loss'], c='deepskyblue', alpha=0.3, label = 'time')
        
        cebra_file_name = f'cebra_pos_hybrid3_model_goal{goal}_ws{window_size}_fold{i+1}.pt'
        cebra_file_path = os.path.join(cebra_model_dir, cebra_file_name)
        cebra_file_path = cebra_file_path.replace("/", "\\")        
        cebra_time3_model = cebra.CEBRA.load(cebra_file_path)  
        ax.plot(cebra_pos_hybrid3_model.state_dict_['loss'], c='deepskyblue', alpha=0.6, label = 'hybrid')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('InfoNCE Loss')
        plt.legend(bbox_to_anchor=(0.5,0.3), frameon = False )
        plt.show()

    fig_name = f"cebra_pos_shuffled3_time_hybrid_loss_goal{goal}_ws{window_size}.png"
    fig_path = os.path.join(fig_dir, fig_name)
    plt.savefig(fig_path)


    ####### CALCULATE r2 SCORES ########


    pass


