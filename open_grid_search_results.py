import glob
import pickle
from cebra_embedding_time_grid_search import CustomCEBRA


def main():

    data_dir = 'D:/analysis/carlas_windowed_data/honeycomb_neural_data/rat_7/6-12-2019/'

    file = glob.glob(data_dir + 'grid_search*.pkl')[0]

    with open(file, 'rb') as f:
        grid_search_results = pickle.load(f)

    print(grid_search_results)

    pass


if __name__ == "__main__":
    main()