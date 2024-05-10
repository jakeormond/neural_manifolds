
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd


# sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
# from utilities.get_directories import get_data_dir


def get_explained_variance(inputs):
    max_n_components = inputs.shape[1]
    pca = PCA(n_components=max_n_components)
    pca.fit(inputs)
    explained_variance = pca.explained_variance_

    return explained_variance


def analyze_screeplot_df(explained_variance_df):

    # find max explained variance for each window size
    max_explained_variance = explained_variance_df.max()

    # find the window size with the max explained variance
    max_window_size = max_explained_variance.idxmax()

    # find the max explained variance
    max_explained_variance_value = max_explained_variance.max()

    print(f'window size with max explained variance: {max_window_size}')

    return max_window_size, max_explained_variance_value



def plot_screeplot(explained_variance_df, spike_dir):

    window_sizes = explained_variance_df.columns
    n_windows = len(window_sizes)
    # n_components is the length of a column in the df
    n_components = len(explained_variance_df.index)

    fig, axs = plt.subplots(1, len(window_sizes), figsize=((n_windows+1) * 5, 5)) 

    # get max values from explained_variance_df
    PC_values = np.arange(n_components) + 1
    max_ev = explained_variance_df.max().max()

    for i, window_size in enumerate(window_sizes):

        explained_variance = explained_variance_df[window_size].values
        axs[i].plot(PC_values, explained_variance, 'o-', linewidth=2, color='blue')
        axs[i].set_title(f'scree plot - window size: {window_size}')
        axs[i].set_xlabel('Principal Component')
        axs[i].set_ylabel('Variance Explained')
        axs[i].set_xticks(np.arange(5, n_components + 1, 10))
        axs[i].set_ylim([0, max_ev+.5])

    plt.show()
    # save the figure
    fig_name = 'scree_plot.png'
    fig_path = os.path.join(spike_dir, fig_name)
    fig.savefig(fig_path)
    plt.close(fig)


def main():
    
    data_dir = 'D:/analysis/carlas_windowed_data/honeycomb_neural_data'
    rat_and_session = ['rat_3/25-3-2019', 'rat_7/6-12-2019', 'rat_8/15-10-2019', 'rat_9/10-12-2021', 'rat_10/23-11-2021']
    n_rats = len(rat_and_session)
    windows_with_max_ev = {}

    window_sizes = [20, 50, 100, 250, 500]

    for rs in rat_and_session:
        spike_dir = os.path.join(data_dir, rs, 'physiology_data')    

        for i, window_size in enumerate(window_sizes): 

            ############ LOAD SPIKE DATA #####################
            # load numpy array of neural data
            inputs_file_name = f'inputs_{window_size}.npy'
            inputs_path = os.path.join(spike_dir, inputs_file_name)
            inputs = np.load(os.path.join(inputs_path))

            explained_variance = get_explained_variance(inputs)

            if i == 0:
                n_components = inputs.shape[1]

                print(f'rat and session: {rs}')
                print(f'number of neurons: {n_components}')

                # create a data frame to hold the explained variance for each window size
                explained_variance_df = pd.DataFrame(columns=window_sizes)

            explained_variance_df[window_size] = explained_variance

        explained_variance_df.to_csv(os.path.join(spike_dir, 'explained_variance.csv'))

        max_window_size, max_value = analyze_screeplot_df(explained_variance_df)
        windows_with_max_ev[rs] = (max_window_size, max_value)

        ############## make figure with horizontal subplots for each window size ###############
        plot_screeplot(explained_variance_df, spike_dir)

    print(windows_with_max_ev)

if __name__ == '__main__':
    #
    main()