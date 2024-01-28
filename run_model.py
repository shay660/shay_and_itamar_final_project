import sys
from os import mkdir, chdir
import datetime
import joblib
import pandas as pd
from main import predict_and_calculate_loss, find_significant_kmers


def main():
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
    model_features = model.feature_names_in_
    find_significant_kmers(model)
    new_columns = pd.DataFrame({feat: [0] for feat in model_features},
                               index=data.index)
    # Select only the columns that do not already exist in the data DataFrame
    new_columns = new_columns.loc[:, ~new_columns.columns.isin(data.columns)]
    # Concatenate the new_columns DataFrame with the original data DataFrame
    data = pd.concat([new_columns, data], axis=1)
    predict_and_calculate_loss(model, data[model.feature_names_in_],
                               data.iloc[:, -3:], name, file)
    file.close()


if __name__ == '__main__':
    main()

