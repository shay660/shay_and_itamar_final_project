from time import time

import pandas as pd

from load_data import load_seq_data
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == '__main__':
    start = time()
    df: pd.DataFrame = load_seq_data("./data/small_data_samples.txt",
                                     "./data/3U.models.3U.00A.seq1022_param.txt",
                                     3, 7)
    end = time()
