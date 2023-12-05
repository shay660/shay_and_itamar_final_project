import datetime
from os import mkdir, chdir
import sys
from time import time
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_data import matrix_generator
from model_generator import linear_reg_model_generator, \
    lasso_and_cross_validation_generator
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load


def load_files(model_to_run: int, path_to_samples: str, path_to_response) -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    """
    load the response vector and the file according to the model_to_run
    :param model_to_run: int that indicates which samples is needed
    :param path_to_samples: path to the samples file.
    :param path_to_response: path to the responses file.
    :return: tuple of samples and the corresponding responses as pandas
    DataFrame.
    """
    f = pd.read_csv(path_to_samples, delimiter='\t', index_col=0, skiprows=2,
                    names=['id', 'seq'])
    if model_to_run == 1 or model_to_run == 3:  # with polA tail.
        responses_with_polyA = pd.read_csv(path_to_response, delimiter='\t',
                                           index_col=0, skiprows=1,
                                           names=['id', 'degradation rate',
                                                  'step_loc', 't0'])
        if model_to_run == 3:  # early-onset (with polyA tail).
            print("************ \nFilters early on-set responses", flush=True)
            early_responses_with_polyA = responses_with_polyA[
                responses_with_polyA['t0'] == 1]
            return f, early_responses_with_polyA
        return f, responses_with_polyA

    else:  # without polyA tail.
        responses_without_polyA = pd.read_csv(path_to_response, delimiter='\t',
                                              index_col=0, skiprows=1,
                                              names=['id', 'degradation rate',
                                                     'step_loc', 't0'])
        if model_to_run == 4:  # early onset data (without polyA tail).
            print("************ \nFilters early on-set responses", flush=True)
            early_responses_without_polyA = responses_without_polyA[
                responses_without_polyA['t0'] == 1]
            return f, early_responses_without_polyA
        return f, responses_without_polyA


def save_or_upload_matrix(to_generate_matrix: bool, model: int,
                          path_to_samples: str, path_to_response: str,
                          name_of_file: str, min_length_kmer: int, max_length_kmer: int) -> pd.DataFrame:
    """
    generate save or upload the samples' matrix, join  with the responses,
    according to the to_generate_matrix argument.
    :param to_generate_matrix:
    :param model: int that indicates which samples is needed
    :param path_to_samples: path to the samples file.
    :param path_to_response: path to the responses file.
    :param name_of_file: The name of new file or the name of the file, to open.
    :return: A upload the samples' matrix, join  with the responses pandas
    DataFrame.
    """
    if to_generate_matrix:
        print("************ \n Parses text file arguments into dataFrames",
              flush=True)
        samples, responses = load_files(model, path_to_samples,
                                        path_to_response)
        print("************ \nJoins samples to responses", flush=True)
        samples_with_responses = matrix_generator(samples, responses,
                                                  min_length_kmer,
                                                  max_length_kmer)
        print("************ \nSaves DataFrames as csv files", flush=True)
        samples_with_responses.to_csv(f"./data/matrices/{name_of_file}", index=True)
        return samples_with_responses

    print("************ \nOpens saved csv files", flush=True)
    return pd.read_csv(f"./data/matrices/{name_of_file}", index_col=0)


def argument_parser(args: List[str], generate_model: bool):
    """
    parse given command line arguments
    :param args: command line arguments
    :param generate_model: boolean flag
    :return: parsed arguments
    """
    _model_to_run = int(args[1])
    _alphas = [float(x) for x in args[2].split(",")]
    _name_of_file = args[3]
    _name_of_model = args[4]
    _path_to_samples = args[5] if generate_model else None
    _path_to_responses = args[6] if generate_model else None
    _min_length_kmer = int(sys.argv[7]) if generate_model else None
    _max_length_kmer = int(sys.argv[8]) if generate_model else None

    return _model_to_run, _alphas, _name_of_file, _name_of_model,\
        _path_to_samples, _path_to_responses, _min_length_kmer, _max_length_kmer


def predict_and_calculate_loss(_model, X_test, y_test, _name_of_model: str,
                               file):
    prediction = _model.predict(X_test)
    mse = mean_squared_error(y_test, prediction[:, 0])
    r = np.round(np.corrcoef(y_test, prediction[:, 0])[0,1],3)
    # ['degradation rate']
    file.write(f"MSE of the Linear regression = {round(mse,3)}\n")
    file.write(f"r of the Linear regression = {r}\n")

    print("******** Save the results ********", flush=True)
    prediction_df = pd.DataFrame(
        {'True_Degradation_Rate': y_test,
         'Predicted_Degradation_Rate': prediction[:, 0]})
    # ['degradation rate']
    prediction_df.to_csv(f"{_name_of_model}_results.csv",
                         index=False)
    # make_scatter_plot(X_test, _model, r, y_test, _name_of_model)
    make_heatmap_plot(prediction_df["True_Degradation_Rate"], prediction_df[
        "Predicted_Degradation_Rate"], r, _name_of_model)


def make_heatmap_plot(X, y, r, _name_of_model):
    heatmap, xedges, yedges = np.histogram2d(X, y, bins=50)

    # Flatten the heatmap to use as colors for the points
    x_bin_indices = np.digitize(X, xedges)
    y_bin_indices = np.digitize(y, yedges)

    # Make sure the indices are within bounds
    x_bin_indices = np.clip(x_bin_indices, 0, heatmap.shape[0] - 1)
    y_bin_indices = np.clip(y_bin_indices, 0, heatmap.shape[1] - 1)

    # Flatten the density matrix to use as colors for the points
    colors = heatmap[x_bin_indices - 1, y_bin_indices - 1]

    # Set the background color to white
    plt.figure(figsize=(8, 6))
    plt.gca().set_facecolor('white')

    # Create a scatter plot with colors based on density
    plt.scatter(X, y, c=colors, cmap='viridis', alpha=0.5, edgecolor='none')
    plt.plot([-5, 1], [-5, 1], color='grey', linestyle='--', label='y=x')

    # Add the r value as text to the plot
    plt.text(0.05, 0.95, f'r = {r:.2f}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.xlabel("Ture Degradation Rate")
    plt.ylabel("Predicted Degradation Rate")
    plt.title(f"{_name_of_model.replace('name', ' ')}")

    # Display or save the plot
    print("******* Save the plot *******")
    plt.savefig(f"{_name_of_model}_plot.png")


if __name__ == '__main__':
    start = time()
    to_generate_model: bool = len(sys.argv) > 5
    model_to_run, alphas, name_of_file, name_of_model, path_to_samples, path_to_responses, min_length_kmer,\
        max_length_kmer = argument_parser(sys.argv, to_generate_model)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory_name = f"./models/{name_of_model}name{timestamp}"
    mkdir(directory_name)

    samples_to_run: pd.DataFrame = save_or_upload_matrix(to_generate_model,
                                                         model_to_run,
                                                         path_to_response=path_to_responses,
                                                         path_to_samples=path_to_samples,
                                                         name_of_file=name_of_file, min_length_kmer= min_length_kmer,
                                                         max_length_kmer= max_length_kmer)
    chdir(directory_name)
    file = open("summary.txt", "w")
    file.write(f"Model {name_of_model}\n")
    file.write(f"Run at {timestamp}\n")
    file.write(f"Given alphas: {alphas}\n")
    print("************  \nRun The model", flush=True)
    model, X_train, y_train = lasso_and_cross_validation_generator(
        samples_to_run, alphas, file)

    predict_and_calculate_loss(model, X_train, y_train, name_of_model, file)
    dump(model, f"{name_of_model}_model.joblib")

    file.close()
    end = time()
    print(f"************ \ntime = {end - start}", flush=True)


