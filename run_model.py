import sys
from os import mkdir, chdir
import datetime
import joblib
import numpy as np
import pandas as pd


def count_kmers_according_to_lasso(df: pd.DataFrame, kmers: np.array) ->pd.DataFrame:
    # Initialize a new DataFrame with the same index as the original DataFrame
    kmer_counts_df = pd.DataFrame(index=df.index)

    # Iterate over each k-mer and count its occurrences in each sequence
    for kmer in kmers:
        kmer_counts_df[kmer] = df['seq'].apply(lambda seq: seq.count(kmer))
    return kmer_counts_df


def main():
    model = joblib.load(sys.argv[1])
    data = pd.read_csv(sys.argv[2], index_col='id')
    name = sys.argv[3]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory_name = f"./models/{name}_{timestamp}"
    mkdir(directory_name)
    chdir(directory_name)
    file = open("summary.txt", "w")
    file.write(f"Model {name}\n")
    file.write(f"Run at {timestamp}\n")
    model_features: np.array = model.feature_names_in_
    samples = count_kmers_according_to_lasso(data, model_features)

    prediction: np.ndarray = model.predict(samples)
    predictions_df = pd.DataFrame(prediction,
                                  columns=['deg rate', 'x0',
                                           't0'])
    predictions_df['id'] = data.index

    # Reorder columns to have 'id' first
    predictions_df = predictions_df[
        ['id', 'deg rate', 'x0', 't0']]
    predictions_df.to_csv(name + ".csv")
    file.close()


if __name__ == '__main__':
    main()

