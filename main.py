import sys
from time import time
from typing import Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

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
                                                  'x0', 't0'])
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
                                                     'x0', 't0'])
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
        samples_with_responses.to_csv(f"./data/matrices/{name_of_file}.csv", index=True)
        return samples_with_responses

    print("************ \nOpens saved csv files", flush=True)
    return pd.read_csv(f"./data/matrices/{name_of_file}.csv", index_col=0)


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


def predict_and_calculate_loss(_model, X_test, y_test):
    prediction = _model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    r = np.corrcoef(y_test['degradation rate'], prediction[:, 0])[0, 1]

    print(f"MSE of the Linear regression = {mse}", flush=True)
    print(f"r of the Linear regression = {r}", flush=True)

    make_scatter_plot(X_test, _model, r, y_test)


def make_scatter_plot(X_test, _model, r, y_test):
    prediction = pd.DataFrame(_model.predict(X_test), index=y_test.index, columns=y_test.columns)
    prediction_vs_true_df = pd.DataFrame({'True degradation rate': y_test['degradation rate'],
                                          'Predicted degradation rate': prediction['degradation rate']})
    fig = px.scatter(prediction_vs_true_df, x='True degradation rate', y='Predicted degradation rate',
                     title='Degradation rate: Predicted vs true', color_discrete_sequence=['red'])
    fig.update_layout(xaxis=dict(showticklabels=False, showline=False),
                      yaxis=dict(showticklabels=False, showline=False))
    fig.add_annotation(text=f"r={r}", x=0.1, y=0.1,
                       showarrow=False, font=dict(size=26, color='black'))
    fig.add_trace(go.Scatter(x=[min(prediction_vs_true_df['True degradation rate']),
                                max(prediction_vs_true_df['True degradation rate'])],
                             y=[min(prediction_vs_true_df['True degradation rate']),
                                max(prediction_vs_true_df['True degradation rate'])],
                             mode='lines',
                             line=dict(color='grey', dash='dash')))
    fig.show()


if __name__ == '__main__':
    start = time()
    to_generate_model: bool = len(sys.argv) > 5
    model_to_run, alphas, name_of_file, name_of_model, path_to_samples, path_to_responses, min_length_kmer,\
        max_length_kmer = argument_parser(sys.argv, to_generate_model)
    samples_to_run: pd.DataFrame = save_or_upload_matrix(to_generate_model,
                                                         model_to_run,
                                                         path_to_response=path_to_responses,
                                                         path_to_samples=path_to_samples,
                                                         name_of_file=name_of_file, min_length_kmer= min_length_kmer,
                                                         max_length_kmer= max_length_kmer)
    print("************  \nRun The model", flush=True)
    model, X_train, y_train = lasso_and_cross_validation_generator(
        samples_to_run, alphas)
    predict_and_calculate_loss(model, X_train, y_train)
    dump(model, f"./models/{name_of_model}.joblib")

    end = time()
    print(f"************ \ntime = {end - start}", flush=True)
