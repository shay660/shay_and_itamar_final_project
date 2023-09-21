import sys
from time import time
from typing import Tuple, List

import joblib
import pandas as pd
from main import predict_and_calculate_loss


if __name__ == '__main__':
    model = joblib.load(sys.argv[1])
    data = pd.read_csv(sys.argv[2])
    name = sys.argv[3]

    predict_and_calculate_loss(model, data[model.feature_names_in_], data.iloc[:, - 3:], _name_of_model=name)

