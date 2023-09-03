import sys
from time import time
from typing import Dict, List

import pandas as pd

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
    print("Split the data to train and test")
    print(f"size of the train set {X_train.shape[0]}")
    print(f"size of the test set {X_test.shape[0]}")

    lasso_model = Lasso(max_iter=5000)
    # set all possible values for the regularization term
    alphas: Dict[str, List[float]] = {'alpha': alphas}

    # cross validation model
    grid_search = GridSearchCV(lasso_model, alphas, cv=5, scoring='neg_mean_squared_error').fit(X_train, y_train)

    print("**** FINISH GRID-SEARCH ****")

    best_alpha = grid_search.best_params_['alpha']
    print(f"Best Alpha: {best_alpha}")

    # chose only the sequences that effects the desegregation rate according to the lasso model.
    lasso_model = grid_search.best_estimator_
    X_train = X_train.loc[:, (lasso_model.coef_ != 0).any(axis=0)]
    X_test = X_test.loc[:, X_train.columns]

    print(f"Number of non-zeroes weightS features after the Lasso: "
          f"{X_train.shape}")

    # creates a linear regression model with the features as the non-zero coefficients of the lasso
    linear_reg_model = LinearRegression().fit(X_train, y_train)
    print(f"**** Linear Regression Fitted ****")

    # estimation
    prediction = linear_reg_model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    print(f"MSE of the Linear regression = {mse}")
    print(f"r2 of the Linear regression = {r2}")


if __name__ == '__main__':
    start = time()

    if len(sys.argv) > 1:
        # reads arguments
        print("************ \n Parses text file arguments into dataFrames")
        f = pd.read_csv(sys.argv[2], delimiter='\t', index_col= 0, skiprows=2, names=['id', 'seq'])
        responses_with_polyA = pd.read_csv(sys.argv[3], delimiter='\t', index_col= 0, skiprows=1, names=['id', 'degradation rate', 'x0', 't0'])
        responses_without_polyA = pd.read_csv(sys.argv[4], delimiter='\t', index_col=0, skiprows=1, names=['id', 'degradation rate', 'x0', 't0'])
        min_length_kmer = int(sys.argv[5])
        max_length_kmer = int(sys.argv[6])

        print("************ \nFilters early on-set responses")
        early_responses_without_polyA = responses_without_polyA[
            responses_without_polyA['t0'] == 1]
        early_responses_with_polyA = responses_with_polyA[
            responses_with_polyA['t0'] == 1]

        print("************ \nJoins samples to responses")
        samples_with_polyA = matrix_generator(f, responses_with_polyA, min_length_kmer, max_length_kmer)
        samples_without_polyA = matrix_generator(f, responses_without_polyA, min_length_kmer, max_length_kmer)
        early_samples_with_polyA = matrix_generator(f, early_responses_with_polyA, min_length_kmer, max_length_kmer)
        early_samples_without_polyA = matrix_generator(f, early_responses_without_polyA, min_length_kmer, max_length_kmer)

        print("************ \nSaves DataFrames as csv files")
        samples_with_polyA.to_csv(f'./data/samples_with_polyA.csv', index= True)
        samples_without_polyA.to_csv(f'./data/samples_without_polyA.csv', index= True)
        early_samples_with_polyA.to_csv(f'./data/early_samples_with_polyA.csv', index=True)
        early_samples_without_polyA.to_csv(f'./data/early_samples_without_polyA.csv', index=True)
    else:
        print("************ \nOpens saved csv files")
        samples_with_polyA = pd.read_csv(f'./data/samples_with_polyA.csv', index_col=0)
        samples_without_polyA = pd.read_csv(f'./data/samples_without_polyA.csv', index_col=0)
        early_samples_with_polyA = pd.read_csv(f'./data/early_samples_with_polyA.csv', index_col=0)
        early_samples_without_polyA = pd.read_csv(f'./data/early_samples_without_polyA.csv', index_col=0)

    print(f"************ \nload data time = {time() - start}")
    #  generate models with all the possible combinations.
    print("************ \nEarly-onset with PolyA")
    model_generator(early_samples_with_polyA, [0.005, 0.008, 0.01])
    print("************ \nEarly-onset without PolyA")
    model_generator(early_samples_without_polyA, [0.005, 0.008, 0.01])
    print("************ \nAll data with PolyA")
    model_generator(samples_with_polyA, [0.005, 0.008, 0.01])
    print("************ \nAll data without PolyA")
    model_generator(samples_without_polyA, [0.005, 0.008, 0.01])
    end = time()
    print(f"************ \ntime = {end - start}")
