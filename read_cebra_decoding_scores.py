import torch
import sys
import os
import pickle
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor
import cebra
import matplotlib.pyplot as plt

sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
from utilities.get_directories import get_data_dir

from cebra_embedding import create_folds


def decode_pos(emb_train, emb_test, label_train, n_neighbors=36):
    pos_decoder = KNeighborsRegressor(n_neighbors, metric = 'cosine')

    pos_decoder.fit(emb_train, label_train)

    pos_pred = pos_decoder.predict(emb_test)

    return pos_pred



if __name__ == "__main__":
    animal = 'Rat46'
    session = '19-02-2024'
    data_dir = get_data_dir(animal, session)

    goal = 52
    window_size = 100

    file_name = f'decoding_scores_goal{goal}_ws{window_size}.pkl'
    file_path = os.path.join(data_dir, file_name)

    with open(file_path, 'rb') as f:
        decoding_scores = pickle.load(f)


    ############ LOAD POSITIONAL DATA ################
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    labels_file_name = f'labels_goal{goal}_ws{window_size}'
    # load numpy array of labels
    labels = np.load(os.path.join(dlc_dir, labels_file_name + '.npy'))
    # keep only the first 2 columns
    labels = labels[:, :2]

    ############ LOAD SPIKE DATA #####################
    # load numpy array of neural data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    inputs_file_name = f'inputs_goal{goal}_ws{window_size}'
    inputs = np.load(os.path.join(spike_dir, inputs_file_name + '.npy'))

    # load convert inputs to torch tensor
    inputs = torch.tensor(inputs, dtype=torch.float32)  

    # will use k-folds with 5 splits
    n_splits = 5
    # kf = KFold(n_splits=n_splits, shuffle=False)
    n_timesteps = inputs.shape[0]
    folds = create_folds(n_timesteps, num_folds=n_splits, num_windows=4)
    test_embeddings = []
    test_labels = []

    for i, (train_index, test_index) in enumerate(folds):

        # load model
        model_name = f'cebra_pos3_model_goal{goal}_ws{window_size}_fold{i+1}.pt'
        model_path = os.path.join(data_dir, model_name)
        cebra_file_path = model_path.replace("/", "\\")  
        cebra_pos3_model = cebra.CEBRA.load(cebra_file_path)

        # get embeddings
        emb_train = cebra_pos3_model.transform(inputs[train_index, :])
        emb_test = cebra_pos3_model.transform(inputs[test_index, :])

        label_train = labels[train_index, :]
        label_test = labels[test_index, :]

        decoded_test_pos = decode_pos(emb_train, emb_test, label_train, n_neighbors=36)

        # plot figure of decoded position vs actual position




        pass




    pass
