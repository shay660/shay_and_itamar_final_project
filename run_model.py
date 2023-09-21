import sys
from time import time
from typing import Tuple, List
from os import mkdir, chdir
import datetime
import joblib
import pandas as pd
from main import predict_and_calculate_loss


if __name__ == '__main__':

    model = joblib.load(sys.argv[1])
    data = pd.read_csv(sys.argv[2])
    name = sys.argv[3]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory_name = f"./models/{name}_{timestamp}"
    mkdir(directory_name)
    chdir(directory_name)
    file = open("summary.txt", "w")
    file.write(f"Model {name}\n")
    file.write(f"Run at {timestamp}\n")

    predict_and_calculate_loss(model, data[model.feature_names_in_], data.iloc[:, - 3:], name, file)
    file.close()

