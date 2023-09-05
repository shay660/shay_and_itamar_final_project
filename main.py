import ast
import sys
from time import time
from typing import Dict, List

import pandas as pd
import enum

from load_data import load_response, matrix_generator
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
                                                        samples.iloc[:, -3:], test_size=0.1,
                                                        random_state=1)
    print("Split the data to train and test", flush= True)
    print(f"size of the train set {X_train.shape[0]}", flush= True)
    print(f"size of the test set {X_test.shape[0]}", flush= True)

    lasso_model = Lasso(max_iter=2500)
    # set all possible values for the regularization term
    alphas: Dict[str, List[float]] = {'alpha': alphas}

    # cross validation model
    grid_search = GridSearchCV(lasso_model, alphas, cv=3, scoring='neg_mean_squared_error', verbose = 2)
    fitted_grid_search = grid_search.fit(X_train, y_train)


    print("**** FINISH GRID-SEARCH ****", flush= True)

    best_alpha = fitted_grid_search.best_params_['alpha']
    print(f"Best Alpha: {best_alpha}")

    # chose only the sequences that effects the desegregation rate according to the lasso model.
    lasso_model = fitted_grid_search.best_estimator_
    X_train = X_train.loc[:, (lasso_model.coef_ != 0).any(axis=0)]
    X_test = X_test.loc[:, X_train.columns]

    print(f"Number of non-zeroes weightS features after the Lasso: "
          f"{X_train.shape}", flush= True)

    # creates a linear regression model with the features as the non-zero coefficients of the lasso
    linear_reg_model = LinearRegression().fit(X_train, y_train)
    print(f"**** Linear Regression Fitted ****", flush= True)

    # estimation
    prediction = linear_reg_model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    print(f"MSE of the Linear regression = {mse}", flush= True)
    print(f"r2 of the Linear regression = {r2}", flush= True)


if __name__ == '__main__':
    start = time()
    print("************ \n Parses text file arguments into dataFrames", flush=True)
    model_to_run = int(sys.argv[1])
    alphas = sys.argv[2]
    if len(sys.argv) > 2:
        # reads arguments
        f = pd.read_csv(sys.argv[3], delimiter='\t', index_col= 0, skiprows=2, names=['id', 'seq'])
        responses_with_polyA = pd.read_csv(sys.argv[4], delimiter='\t', index_col= 0, skiprows=1, names=['id', 'degradation rate', 'x0', 't0'])
        responses_without_polyA = pd.read_csv(sys.argv[5], delimiter='\t', index_col=0, skiprows=1, names=['id', 'degradation rate', 'x0', 't0'])
        min_length_kmer = int(sys.argv[6])
        max_length_kmer = int(sys.argv[7])

        print("************ \nFilters early on-set responses", flush= True)
        early_responses_without_polyA = responses_without_polyA[
            responses_without_polyA['t0'] == 1]
        early_responses_with_polyA = responses_with_polyA[
            responses_with_polyA['t0'] == 1]

        print("************ \nJoins samples to responses", flush= True)
        samples_with_polyA = matrix_generator(f, responses_with_polyA, min_length_kmer, max_length_kmer)
        samples_without_polyA = matrix_generator(f, responses_without_polyA, min_length_kmer, max_length_kmer)
        early_samples_with_polyA = matrix_generator(f, early_responses_with_polyA, min_length_kmer, max_length_kmer)
        early_samples_without_polyA = matrix_generator(f, early_responses_without_polyA, min_length_kmer, max_length_kmer)

        print("************ \nSaves DataFrames as csv files", flush= True)
        samples_with_polyA.to_csv(f'./data/samples_with_polyA.csv', index= True)
        samples_without_polyA.to_csv(f'./data/samples_without_polyA.csv', index= True)
        early_samples_with_polyA.to_csv(f'./data/early_samples_with_polyA.csv', index=True)
        early_samples_without_polyA.to_csv(f'./data/early_samples_without_polyA.csv', index=True)
    else:
        print("************ \nOpens saved csv files", flush= True)
        samples_with_polyA = pd.read_csv(f'./data/samples_with_polyA.csv', index_col=0)
        samples_without_polyA = pd.read_csv(f'./data/samples_without_polyA.csv', index_col=0)
        early_samples_with_polyA = pd.read_csv(f'./data/early_samples_with_polyA.csv', index_col=0)
        early_samples_without_polyA = pd.read_csv(f'./data/early_samples_without_polyA.csv', index_col=0)

    print(f"************ \nload data time = {time() - start}", flush= True)
    #  generate models with all the possible combinations.
    if model_to_run == 1:
        print("************ \nAll data with PolyA", flush= True)
        model_generator(samples_with_polyA, [0.005, 0.008, 0.01])

    elif model_to_run == 2:
        print("************ \nAll data without PolyA", flush= True)
        model_generator(samples_without_polyA, [0.005, 0.008, 0.01])

    elif model_to_run == 3:
        print("************ \nEarly-onset with PolyA", flush= True)
        model_generator(early_samples_with_polyA, [0.005, 0.008, 0.01])

    elif model_to_run == 4:
        print("************ \nEarly-onset without PolyA", flush= True)
        model_generator(early_samples_without_polyA, [0.005, 0.008, 0.01])

    end = time()
    print(f"************ \ntime = {end - start}", flush= True)
