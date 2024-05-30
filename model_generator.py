from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def linear_reg_model_generator(samples: pd.DataFrame, file) -> Tuple[object,
pd.DataFrame, pd.Series]:
    """
    fit a linear regression model with samples
    """
    # X_train, y_train, X_test, y_test = _split_to_train_and_test(samples, file)
    X_train, y_train = samples.iloc[:, :-3], samples.iloc[:, -3:]
    # file.write(f"Number of samples in the train set: {X_train.shape[0]}\n")
    linear_reg_model = LinearRegression().fit(X_train, y_train)
    print(f"**** Linear Regression Fitted ****", flush=True)
    joblib.dump(linear_reg_model, file)
    return linear_reg_model, X_train, y_train
    # return linear_reg_model, X_test, y_test


def _split_to_train_and_test(samples: pd.DataFrame, file):
    """ split the data to train and test set and for samples and responses
    vector. It's also prints the progress of the process."""
    X_train, X_test, y_train, y_test = train_test_split(samples.iloc[:, :-3],
                                                        samples.iloc[:, -3:],
                                                        test_size=0.1,
                                                        random_state=1)
    print("Split the data to train and test", flush=True)
    print(f"size of the train set {X_train.shape[0]}", flush=True)
    file.write(f"Number of samples in the train set: {X_train.shape[0]}\n")
    print(f"size of the test set {X_test.shape[0]}", flush=True)
    file.write(f"Number of samples in the test set: {X_test.shape[0]}\n")
    return X_train, y_train, X_test, y_test


def lasso_and_cross_validation_generator(samples: pd.DataFrame, alphas: list,
                                         file) \
        -> Tuple[LinearRegression, pd.DataFrame, pd.Series]:
    """
    Generate and evaluate a predictive model using Lasso regression followed by
    Linear Regression.
    :param file: summary file
    :param samples: Input data
    :param alphas: List of alpha values for Lasso regularization
    :return: None
    """
    # X_train, y_train, X_test, y_test = _split_to_train_and_test(samples, file)
    X_train, y_train = samples.iloc[:, :-3], samples.iloc[:, -3:]
    file.write(f"Number of samples in the train set: {X_train.shape[0]}\n")
    lasso_model = Lasso(max_iter=2500, tol=3e-3)
    # set all possible values for the regularization term
    alphas: Dict[str, List[float]] = {'alpha': alphas}

    # cross validation model
    folds, n_jobs = 10, 4
    file.write(f"Number of folds: {folds} (n_jobs = {n_jobs})\n")
    grid_search = GridSearchCV(lasso_model, alphas, cv=folds, n_jobs=4,
                               scoring='neg_mean_squared_error', verbose=3)
    fitted_grid_search = grid_search.fit(X_train, y_train)
    print("**** FINISH GRID-SEARCH ****", flush=True)
    best_alpha = fitted_grid_search.best_params_['alpha']
    print(f"Best Alpha: {best_alpha}")
    file.write(f"Best Alpha: {best_alpha}\n")

    # chose only the sequences that effects the desegregation rate according to
    # the lasso model.
    lasso_model = fitted_grid_search.best_estimator_
    X_train = X_train.loc[:, (lasso_model.coef_ != 0).any(axis=0)]
    # X_test = X_test.loc[:, X_train.columns]
    n_features: int = X_train.shape[1]
    print(f"Number of non-zeroes weights features after the Lasso: "
          f"{n_features}", flush=True)
    file.write(f"Number of features used by the Lasso: {n_features}\n")
    # creates a linear regression model with the features as the non-zero
    # coefficients of the lasso
    linear_reg_model: LinearRegression = LinearRegression()
    linear_reg_model.fit(X_train, y_train)
    print(f"**** Linear Regression Fitted ****", flush=True)

    return linear_reg_model, X_train, y_train  # X_test, y_test
