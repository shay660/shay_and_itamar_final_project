import argparse
import datetime
from os import mkdir, chdir
from time import time
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
    file_type: str = path_to_response.split('.')[-1]
    delimiter: str = ',' if file_type == 'csv' else '\t'
    f = pd.read_csv(path_to_samples, index_col=0, skiprows=2, delimiter=delimiter,
                    names=['id', 'seq'])
    responses = load_responses(model_to_run, path_to_response)
    return f, responses


def load_responses(model_to_run, path_to_response):
    file_type: str = path_to_response.split('.')[-1]
    delimiter: str = ',' if file_type == 'csv' else '\t'
    responses: pd.DataFrame = pd.read_csv(path_to_response, delimiter=delimiter,
                                          index_col=0, usecols=['id', 'dg',
                                                                'x0', 't0'])
    responses.columns = ['degradation rate', 'step_loc', 't0']
    if model_to_run == 3 or model_to_run == 4:  # early onset.
        print("************ \nFilters early on-set responses", flush=True)
        responses = responses[responses['t0'] == 1]
    return responses


def save_or_upload_matrix(to_generate_matrix: bool, model: int,
                          path_to_samples: str, path_to_response: str,
                          name_of_file: str, min_length_kmer: int,
                          max_length_kmer: int) -> pd.DataFrame:
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
        samples_with_responses.to_csv(f"./data/matrices/{name_of_file}",
                                      index=True)
        return samples_with_responses

    print("************ \nOpens saved csv files", flush=True)
    return pd.read_csv(f"./data/matrices/{name_of_file}", index_col=0)


def predict_and_calculate_loss(_model, X_test, y_test, _name_of_model: str,
                               file) -> None:
    prediction = _model.predict(X_test)
    mse = mean_squared_error(y_test['degradation rate'], prediction[:, 0])
    r = np.round(
        np.corrcoef(y_test['degradation rate'], prediction[:, 0])[0, 1], 3)

    file.write(f"MSE of the Linear regression = {round(mse, 3)}\n")
    file.write(f"r of the Linear regression = {r}\n")

    print("******** Save the results ********", flush=True)
    prediction_df = pd.DataFrame(
        {'True_Degradation_Rate': y_test['degradation rate'],
         'Predicted_Degradation_Rate': prediction[:, 0]})
    prediction_df.to_csv(f"{_name_of_model}_results.csv",
                         index=False)
    make_heatmap_plot(prediction_df["True_Degradation_Rate"], prediction_df[
        "Predicted_Degradation_Rate"], r, _name_of_model)


def make_heatmap_plot(X, y, r, _name_of_model: str):
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
    plt.text(0.05, 0.95, f'r = {r:.2f}', transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.xlabel("Ture Degradation Rate")
    plt.ylabel("Predicted Degradation Rate")
    plt.title(f"{_name_of_model.replace('name', ' ')}")

    # Display or save the plot
    print("******* Save the plot *******")
    plt.savefig(f"{_name_of_model}_plot.png")


def argument_parser():
    """
    parse given command line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description="The script design to predict "
                                                 "degradation rate of mRNA in the cell "
                                                 "according to it's sequence "
                                                 "using linear regression model.")

    parser.add_argument('--model_to_run', type=int,
                        help="model_to_run - int from 1-4 such that:"
                             "1 - late-onset with polyA tail model."
                             "2 - late-onset without polyA tail model."
                             "3 - early-onset with polyA tail model."
                             "4 - early-onset without polyA tail model.")
    parser.add_argument('--alphas', type=lambda arg: list(map(float,
                                                              arg.split(','))),
                        help="alphas for the lass, separated by commas")
    parser.add_argument('--name_of_matrix', type=str,
                        default=argparse.SUPPRESS,
                        help="A name by which you will read the Kmer matrix "
                             "or open an existing one")
    parser.add_argument('--name_of_model', type=str, default=argparse.SUPPRESS,
                        help="Name of the new model")
    parser.add_argument('--path_to_samples', type=str, default=None,
                        help="Path to samples (sequences) file.")
    parser.add_argument('--path_to_responses', type=str, default=None,
                        help="Path to responses file.")
    parser.add_argument('--min_length_kmer', type=int, default=None,
                        help="The length of the min kmer to look for.")
    parser.add_argument('--max_length_kmer', type=int, default=None,
                        help="The length of the max kmer to look for.")
    parser.add_argument('--to_generate_matrix', type=bool, default=False,
                        help="If True generate a new matrix")
    return parser.parse_args()


def find_significant_kmers(model: LinearRegression) -> None:
    """
    Identify and save the most significant features (columns) based on a trained
    linear regression model.
    :param model: (LinearRegression): Trained linear regression model.
    Saves:
    - A CSV file named "most_significant_kmers" containing the most significant
     features.
    """
    coefficients = model.coef_
    weights_norms = np.linalg.norm(coefficients, axis=0)
    columns_names = model.feature_names_in_

    coefficients_df = pd.DataFrame(
        {'kmers': columns_names, 'weights_norms': weights_norms})

    sorted_coefficients_df = coefficients_df.sort_values(
        by='weights_norms', ascending=False)

    # Extract the most significant features (columns)
    sorted_coefficients_df.to_csv("most_significant_kmers.csv")


def main():
    start = time()

    args = argument_parser()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory_name = f"./models/{args.name_of_model}_{timestamp}"
    mkdir(directory_name)

    samples_to_run: pd.DataFrame = save_or_upload_matrix(
        args.to_generate_matrix,
        args.model_to_run,
        path_to_response=args.path_to_responses,
        path_to_samples=args.path_to_samples,
        name_of_file=args.name_of_matrix,
        min_length_kmer=args.min_length_kmer,
        max_length_kmer=args.max_length_kmer)

    chdir(directory_name)
    file = open("summary.txt", "w")
    file.write(f"Model {args.name_of_model}\n")
    file.write(f"Run at {timestamp}\n")
    file.write(f"Given alphas: {args.alphas}\n")
    print("************  \nRun The model", flush=True)
    model, X_train, y_train = lasso_and_cross_validation_generator(
        samples_to_run, args.alphas, file)

    predict_and_calculate_loss(model, X_train, y_train, args.name_of_model,
                               file)
    find_significant_kmers(model)

    dump(model, f"{args.name_of_model}_model.joblib")

    file.close()
    end = time()
    print(f"************ \ntime = {end - start}", flush=True)


if __name__ == '__main__':
    main()
