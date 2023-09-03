import sys
from time import time
from typing import Dict, List

import pandas as pd

from load_data import load_response, _matrix_generator
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


def model_generator(samples: pd.DataFrame, response_vec: pd.DataFrame, alphas: list) -> None:
    """
    Generate and evaluate a predictive model using Lasso regression followed by Linear Regression.
    :param samples: Input data
    :param response_vec: responses data
    :param alphas: List of alpha values for Lasso regularization
    :return: None
    """

    # merge samples with responses iff we have the needed data for both. many samples are filtered because
    # we don't have a proper response for them
    samples = samples.join(response_vec, how="inner")
    X_train, X_test, y_train, y_test = train_test_split(samples.iloc[:, :-3],samples.iloc[:, -3:], test_size=0.1,
                                                        random_state=1)

    lasso_model = Lasso(max_iter=5000)
    # set all possible values for the regularization term
    alphas: Dict[str, List[float]] = {'alpha': alphas}

    # cross validation model
    grid_search = GridSearchCV(lasso_model, alphas, cv=10, scoring='neg_mean_squared_error').fit(X_train, y_train)

    best_alpha = grid_search.best_params_['alpha']
    print(f"Best Alpha: {best_alpha}")

    # chose only the sequences that effects the desegregation rate according to the lasso model.
    lasso_model = grid_search.best_estimator_
    X_train = X_train.loc[:, (lasso_model.coef_ != 0).any(axis=0)]
    X_test = X_test.loc[:, X_train.columns]

    # creates a linear regression model with the features as the non-zero coefficients of the lasso
    linear_reg_model = LinearRegression().fit(X_train, y_train)

    # estimation
    prediction = linear_reg_model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    print(f"MSE of the Linear regression = {mse}")
    print(f"r2 of the Linear regression = {r2}")


if __name__ == '__main__':
    start = time()

    # reads arguments
    f: list = open(sys.argv[1], "r").readlines()
    responses_with_polyA = load_response(sys.argv[2])
    responses_without_polyA = load_response(
        sys.argv[3])

    # filter samples with t0=1 to be the early on-set vectors
    early_responses_without_polyA = responses_without_polyA[
        responses_without_polyA['t0'] == '1']
    early_responses_with_polyA = responses_with_polyA[
        responses_with_polyA['t0'] == '1']

    # creates 4 dataframes
    samples_with_polyA = _matrix_generator(f, responses_with_polyA, 3, 7)
    samples_without_polyA = _matrix_generator(f,responses_without_polyA, 3, 7)
    early_samples_with_polyA = _matrix_generator(f,early_responses_with_polyA, 3, 7)
    early_samples_without_polyA = _matrix_generator(f,early_responses_without_polyA, 3, 7)

    print(f"load data time = {time() - start}")

    #  generate models with all the possible combinations.
    print("Early-onset with PolyA")
    model_generator(early_samples_with_polyA, early_responses_with_polyA,[0.005, 0.008, 0.01])
    print("**********************************")
    print("Early-onset without PolyA")
    model_generator(early_samples_without_polyA, early_responses_without_polyA,[0.005, 0.008, 0.01])
    print("**********************************")
    print("All data with PolyA")
    model_generator(samples_with_polyA, responses_with_polyA, [0.005, 0.008,0.01])
    print("**********************************")
    print("All data without PolyA")
    model_generator(samples_without_polyA, responses_without_polyA, [0.005,0.008, 0.01])
    print("**********************************")
    end = time()
    print(f"time = {end - start}")
