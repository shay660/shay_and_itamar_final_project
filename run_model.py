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
    """
    Main function that loads a pre-trained Lasso model and input data,
    counts k-mers in the sequences according to model features,
    generates predictions, and saves the results.

    Steps:
        1. Load the Lasso model from the specified file.
        2. Read the input data (CSV format) with sequences indexed by 'id'.
        3. Count the occurrences of k-mers (as per model features) in the input data sequences.
        4. Generate predictions using the model based on k-mer counts.
        5. Save the predictions to a CSV file, with an additional summary in a text file.

    Usage:
        $ python script.py <model_file> <data_file> <output_name>

        - model_file: Path to the pre-trained Lasso model (joblib format
        version 1.3.2).
        - data_file: Path to the CSV file containing the sequence data.
        - output_name: Name used for the output directory and result files.
    """
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

