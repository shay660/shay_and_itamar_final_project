from time import time
from typing import Dict, List

import pandas as pd

from load_data import load_seq_data
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


def model_generator(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-3],
                                                        df.iloc[:, -3:],
                                                        test_size=0.1,
                                                        random_state=1)
    lasso_model = Lasso(max_iter=2500)
    alphas: Dict[str, List[float]] = {'alpha': [0.01, 0.02, 0.015, 0.008]}
    grid_search = GridSearchCV(lasso_model, alphas, cv=5,
                               scoring='neg_mean_squared_error').fit(X_train,
                                                                     y_train)
    best_alpha = grid_search.best_params_['alpha']
    best_score = -grid_search.best_score_  # Convert back to positive
    print(f"Best Alpha: {best_alpha}")
    print(f"Best Score: {best_score:.4f}")
    lasso_model = grid_search.best_estimator_
    # chose only the sequences that effects the desegregation rate according
    # to the lasso model.
    X_train = X_train.loc[:, (lasso_model.coef_ != 0).any(axis= 0)]
    X_test = X_test.loc[:, X_train.columns]
    linear_reg_model = LinearRegression().fit(X_train, y_train)
    prediction = linear_reg_model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    # r2 = r2_score(y_test, prediction)
    end = time()
    print(f"time = {end - start}")
    # print(lasso_model.coef_)
    print(f"MSE of the Linear regression = {mse}")
    # print(f"r2 of the Linear regression = {r2}")


if __name__ == '__main__':
    start = time()
    early_onset_df, late_onset_df = load_seq_data("./data/1000_samples_data.txt",
                                        "./data/3U.models.3U.40A.seq1022_param.txt",
                                     3, 7)

    model_generator(early_onset_df)

