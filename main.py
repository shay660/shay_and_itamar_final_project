import ast
import sys
from pyexpat import model
from time import time
from typing import Dict, List, Tuple

import pandas as pd
import enum

from load_data import matrix_generator
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


def model_generator(samples: pd.DataFrame, alphas: list) -> None:
    """
    Generate and evaluate a predictive model using Lasso regression followed by Linear Regression.
    :param samples: Input data
    :param response_vec: responses data
    :param alphas: List of alpha values for Lasso regularization
    :return: None
    """

    # merge samples with responses iff we have the needed data for both. many samples are filtered because
    # we don't have a proper response for them
    X_train, X_test, y_train, y_test = train_test_split(samples.iloc[:, :-3],
                                                        samples.iloc[:, -3:],
                                                        test_size=0.1,
                                                        random_state=1)
    print("Split the data to train and test", flush=True)
    print(f"size of the train set {X_train.shape[0]}", flush=True)
    print(f"size of the test set {X_test.shape[0]}", flush=True)

    lasso_model = Lasso(max_iter=2500)
    # set all possible values for the regularization term
    alphas: Dict[str, List[float]] = {'alpha': alphas}

    # cross validation model
    grid_search = GridSearchCV(lasso_model, alphas, cv=3,
                               scoring='neg_mean_squared_error', verbose=3)
    fitted_grid_search = grid_search.fit(X_train, y_train)

    print("**** FINISH GRID-SEARCH ****", flush=True)

    best_alpha = fitted_grid_search.best_params_['alpha']
    print(f"Best Alpha: {best_alpha}")

    # chose only the sequences that effects the desegregation rate according to the lasso model.
    lasso_model = fitted_grid_search.best_estimator_
    X_train = X_train.loc[:, (lasso_model.coef_ != 0).any(axis=0)]
    X_test = X_test.loc[:, X_train.columns]

    print(f"Number of non-zeroes weightS features after the Lasso: "
          f"{X_train.shape}", flush=True)

    # creates a linear regression model with the features as the non-zero coefficients of the lasso
    linear_reg_model = LinearRegression().fit(X_train, y_train)
    print(f"**** Linear Regression Fitted ****", flush=True)

    # estimation
    prediction = linear_reg_model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    print(f"MSE of the Linear regression = {mse}", flush=True)
    print(f"r2 of the Linear regression = {r2}", flush=True)


def load_files(model_to_run: int, path_to_samples: str, path_to_response) -> \
                                        Tuple[pd.DataFrame, pd.DataFrame]:
    """

    :param model_to_run:
    :param path_to_samples:
    :param path_to_response:
    :return:
    """
    f = pd.read_csv(path_to_samples, delimiter='\t', index_col=0, skiprows=2,
                    names=['id', 'seq'])
    if model_to_run == 1 or model_to_run == 3:
        responses_with_polyA = pd.read_csv(path_to_response, delimiter='\t',
                                           index_col=0, skiprows=1,
                                           names=['id', 'degradation rate',
                                                  'x0', 't0'])
        if model_to_run == 3:
            print("************ \nFilters early on-set responses", flush=True)
            early_responses_with_polyA = responses_with_polyA[
                responses_with_polyA['t0'] == 1]
            return f, early_responses_with_polyA
        return f, responses_with_polyA

    else:
        responses_without_polyA = pd.read_csv(path_to_response, delimiter='\t',
                                              index_col=0, skiprows=1,
                                              names=['id', 'degradation rate',
                                                     'x0', 't0'])
        if model_to_run == 4:
            print("************ \nFilters early on-set responses", flush=True)
            early_responses_without_polyA = responses_without_polyA[
                responses_without_polyA['t0'] == 1]
            return f, early_responses_without_polyA
        return f, responses_without_polyA


def save_or_upload_matrix(to_generate_matrix: bool, model: int,
                          path_to_samples: str, path_to_response: str,
                          name_of_file: str) -> pd.DataFrame:
    """

    :param to_generate_matrix:
    :param model:
    :param path_to_samples:
    :param path_to_response:
    :param name_of_file:
    :return:
    """
    if to_generate_matrix:
        samples, responses = load_files(model, path_to_samples,
                                        path_to_response)
        min_length_kmer = int(sys.argv[6])
        max_length_kmer = int(sys.argv[7])
        print("************ \nJoins samples to responses", flush=True)
        samples_with_responses = matrix_generator(samples, responses,
                                                  min_length_kmer,
                                                  max_length_kmer)
        print("************ \nSaves DataFrames as csv files", flush=True)
        samples_with_responses.to_csv(f"./data/{name_of_file}", index=True)
        return samples_with_responses

    print("************ \nOpens saved csv files", flush=True)
    return pd.read_csv(f"./data/{name_of_file}", index_col=0)


if __name__ == '__main__':
    start = time()
    print("************ \n Parses text file arguments into dataFrames",
          flush=True)
    to_generate_model: bool = len(sys.argv) > 4
    model_to_run = int(sys.argv[1])
    alphas = [float(x) for x in sys.argv[2][1:-1].split(",")]
    path_to_samples = sys.argv[4] if to_generate_model else None
    path_to_responses = sys.argv[5] if to_generate_model else None
    name_of_file = sys.argv[3]

    samples_to_run: pd.DataFrame = save_or_upload_matrix(to_generate_model,
                                                         model_to_run,
                                                         path_to_response=path_to_responses,
                                                         path_to_samples=path_to_samples,
                                                         name_of_file=name_of_file)
    print("************  \nRum The model", flush=True)
    model_generator(samples_to_run, alphas)
    # if len(sys.argv) > 2:
    #     # reads arguments
    #     f = pd.read_csv(sys.argv[3], delimiter='\t', index_col=0, skiprows=2,
    #                     names=['id', 'seq'])
    #     responses_with_polyA = pd.read_csv(sys.argv[4], delimiter='\t',
    #                                        index_col=0, skiprows=1,
    #                                        names=['id', 'degradation rate',
    #                                               'x0', 't0'])
    #     responses_without_polyA = pd.read_csv(sys.argv[5], delimiter='\t',
    #                                           index_col=0, skiprows=1,
    #                                           names=['id', 'degradation rate',
    #                                                  'x0', 't0'])
    #     min_length_kmer = int(sys.argv[6])
    #     max_length_kmer = int(sys.argv[7])
    #
    #     print("************ \nFilters early on-set responses", flush=True)
    #     early_responses_with_polyA = responses_with_polyA[
    #         responses_with_polyA['t0'] == 1]
    #     early_responses_without_polyA = responses_without_polyA[
    #         responses_without_polyA['t0'] == 1]
    #
    #     print("************ \nJoins samples to responses", flush=True)
    #     samples_with_polyA = matrix_generator(f, responses_with_polyA,
    #                                           min_length_kmer, max_length_kmer)
    #     samples_without_polyA = matrix_generator(f, responses_without_polyA,
    #                                              min_length_kmer,
    #                                              max_length_kmer)
    #     early_samples_with_polyA = matrix_generator(f,
    #                                                 early_responses_with_polyA,
    #                                                 min_length_kmer,
    #                                                 max_length_kmer)
    #     early_samples_without_polyA = matrix_generator(f,
    #                                                    early_responses_without_polyA,
    #                                                    min_length_kmer,
    #                                                    max_length_kmer)
    #
    #     print("************ \nSaves DataFrames as csv files", flush=True)
    #     samples_with_polyA.to_csv(f'./data/samples_with_polyA.csv', index=True)
    #     samples_without_polyA.to_csv(f'./data/samples_without_polyA.csv',
    #                                  index=True)
    #     early_samples_with_polyA.to_csv(f'./data/early_samples_with_polyA.csv',
    #                                     index=True)
    #     early_samples_without_polyA.to_csv(
    #         f'./data/early_samples_without_polyA.csv', index=True)
    # else:
    #     print("************ \nOpens saved csv files", flush=True)
    #     samples_with_polyA = pd.read_csv(f'./data/samples_with_polyA.csv',
    #                                      index_col=0)
    #     samples_without_polyA = pd.read_csv(f'./data/samples_without_polyA.csv',
    #                                         index_col=0)
    #     early_samples_with_polyA = pd.read_csv(
    #         f'./data/early_samples_with_polyA.csv', index_col=0)
    #     early_samples_without_polyA = pd.read_csv(
    #         f'./data/early_samples_without_polyA.csv', index_col=0)
    #
    # print(f"************ \nload data time = {time() - start}", flush=True)
    # #  generate models with all the possible combinations.
    # if model_to_run == 1:
    #     print("************ \nAll data with PolyA", flush=True)
    #     model_generator(samples_with_polyA, [0.005, 0.008, 0.01])
    #
    # elif model_to_run == 2:
    #     print("************ \nAll data without PolyA", flush=True)
    #     model_generator(samples_without_polyA, [0.005, 0.008, 0.01])
    #
    # elif model_to_run == 3:
    #     print("************ \nEarly-onset with PolyA", flush=True)
    #     model_generator(early_samples_with_polyA, [0.005, 0.008, 0.01])
    #
    # elif model_to_run == 4:
    #     print("************ \nEarly-onset without PolyA", flush=True)
    #     model_generator(early_samples_without_polyA, [0.005, 0.008, 0.01])

    end = time()
    print(f"************ \ntime = {end - start}", flush=True)
