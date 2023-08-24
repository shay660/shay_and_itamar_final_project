from time import time

import pandas as pd
import sklearn.linear_model

from load_data import load_seq_data
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == '__main__':
    start = time()
    df: pd.DataFrame = load_seq_data("./data/15k.txt",
                                     "./data/3U.models.3U.00A.seq1022_param.txt",
                                     3, 7)
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1],
                                                        df.iloc[:, -1],
                                                        test_size=0.1,
                                                        random_state=1)
    model: sklearn.linear_model.Lasso = Lasso(alpha=0.01).fit(X_train, y_train)
    prediction = model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)
    end = time()
    print(model.coef_)
    print(f"MSE = {mse}")
    print(f"r2 = {r2}")
    print(end-start)
    print(prediction)
