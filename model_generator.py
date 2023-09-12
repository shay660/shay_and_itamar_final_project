from typing import Dict, List, Tuple

import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def linear_reg_model_generator(samples: pd.DataFrame) -> Tuple[object,
                                                       pd.DataFrame, pd.Series]:
    """
    fit a linear regression model with samples
    """
    # X_train, y_train, X_test, y_test = _split_to_train_and_test(samples)
    X_train, y_train = samples.iloc[:, :-3], samples.iloc[:, -3:]
    linear_reg_model = LinearRegression().fit(X_train, y_train)
    print(f"**** Linear Regression Fitted ****", flush=True)

    return linear_reg_model, X_train, y_train


def _split_to_train_and_test(samples: pd.DataFrame):
    """ split the data to train and test set and for samples and responses
    vector. It's also prints the progress of the process."""
    X_train, X_test, y_train, y_test = train_test_split(samples.iloc[:, :-3],
                                                        samples.iloc[:, -3:],
                                                        test_size=0.0,
                                                        random_state=1)
    print("Split the data to train and test", flush=True)
    print(f"size of the train set {X_train.shape[0]}", flush=True)
    print(f"size of the test set {X_test.shape[0]}", flush=True)
    return X_train, y_train, X_test, y_test


def lasso_and_cross_validation_generator(samples: pd.DataFrame, alphas: list) \
        -> Tuple[object, pd.DataFrame, pd.Series]:
    """
    Generate and evaluate a predictive model using Lasso regression followed by
    Linear Regression.
    :param samples: Input data
    :param alphas: List of alpha values for Lasso regularization
    :return: None
    """
    # X_train, y_train, X_test, y_test = _split_to_train_and_test(samples)
    X_train, y_train = samples.iloc[:, :-3], samples.iloc[:, -3:]
    lasso_model = Lasso(max_iter=2500, tol=3e-3)
    # set all possible values for the regularization term
    alphas: Dict[str, List[float]] = {'alpha': alphas}

    # cross validation model
    grid_search = GridSearchCV(lasso_model, alphas, cv=8, n_jobs=5,
                               scoring='neg_mean_squared_error', verbose=3)
    fitted_grid_search = grid_search.fit(X_train, y_train)

    print("**** FINISH GRID-SEARCH ****", flush=True)

    best_alpha = fitted_grid_search.best_params_['alpha']
    print(f"Best Alpha: {best_alpha}")

    # chose only the sequences that effects the desegregation rate according to
    # the lasso model.
    lasso_model = fitted_grid_search.best_estimator_
    X_train = X_train.loc[:, (lasso_model.coef_ != 0).any(axis=0)]
    # X_test = X_test.loc[:, X_train.columns]

    print(f"Number of non-zeroes weightS features after the Lasso: "
          f"{X_train.shape[1]}", flush=True)

    # creates a linear regression model with the features as the non-zero
    # coefficients of the lasso
    linear_reg_model = LinearRegression().fit(X_train, y_train)
    print(f"**** Linear Regression Fitted ****", flush=True)

    return linear_reg_model, X_train, y_train  # X_test, y_test


