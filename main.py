from time import time

import pandas as pd
import sklearn.linear_model

from load_data import load_seq_data
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    start = time()
    df: pd.DataFrame = load_seq_data("./data/15k_samples_data.txt",
                                     "./data/3U.models.3U.00A.seq1022_param.txt",
                                     3, 7)
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1],
                                                        df.iloc[:, -2:],
                                                        test_size=0.1,
                                                        random_state=1)
    # model: sklearn.linear_model.Lasso = Lasso(alpha=0.1).fit(X_train, y_train)

    model: sklearn.linear_model.Lasso = Lasso()

    alphas = {'alpha': [0.001, 0.003, 0.007, 0.01]}
    grid_search = GridSearchCV(model, alphas, cv=5, scoring='neg_mean_squared_error').fit(X_train, y_train)
    best_alpha = grid_search.best_params_['alpha']
    best_score = -grid_search.best_score_  # Convert back to positive

    print(f"Best Alpha: {best_alpha}")
    print(f"Best Score: {best_score:.4f}")

    model = grid_search.best_estimator_

    prediction = model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)
    end = time()
    print(f"time = {end - start}")
    # print(model.coef_)
    print(f"MSE = {mse}")
    print(f"r2 = {r2}")

