import itertools
from typing import IO, Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm


def load_seq_data(samples_path: str, response_path: str, min_length: int,
                  max_length: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    loads the data from a text file into data frame that count how many times
    each sequence in the length of 3-7 nucleotides is in each 3'URR.
    :param path: path to a text file
    :return: data frame that count how many times each sequence in the length of
     3-7 nucleotides is in each 3'URR. The shape (n_genes, 4^3+4^4+4^5+4^6+4^7)
    """
    f: list = open(samples_path, "r").readlines()

    # loads the responses vector
    early_onset_responses_vec, late_onset_responses_vec = load_response(response_path)

    early_onset_matrix = matrix_generator(f, max_length, min_length, early_onset_responses_vec)
    late_onset_matrix = matrix_generator(f, max_length, min_length, late_onset_responses_vec)
    return early_onset_matrix, late_onset_matrix

def matrix_generator(f, max_length, min_length, response_vec):
    # create dict. key is id, value is a dict where the key is k_mer and the
    # value is its frequency in the id seq.
    k_mers_counter: Dict[str: Dict[str, int]] = {}
    for line in tqdm(f[2:]):
        line = line.split()
        id, seq = line[0], line[1]
        # checks if we have the response value for the id before loading it to the dataset
        if id in response_vec.index.tolist():
            k_mers_counter[id] = k_mers_count(seq, min_length, max_length)
    samples = pd.DataFrame.from_dict(k_mers_counter, orient="index", dtype=int)
    samples.fillna(0, inplace=True)
    df: pd.DataFrame = samples.join(response_vec, how="inner")
    return df


def k_mers_count(seq: str, min_length: int, max_length: int) -> Dict[str, int]:
    """
    counts the number of times each k_mers in the length of min_length to
    max_length in a sequence.
    :param min_length: the min length of k_mer that is look for.
    :param max_length: the max length of k_kmer that is look for.
    :return: A dict which the key is the k_mer and the value is the number of
    times that the k_mer is in the seq.
    """
    # Start with an empty dictionary
    counts = {}
    for k in range(min_length, max_length + 1):
        # Calculate how many kmers of length k there are
        num_kmers = len(seq) - k + 1
        # Loop over the kmer start positions
        for i in range(num_kmers):
            # Slice the string to get the kmer
            kmer = seq[i:i + k]
            # Increment the count for this kmer
            if kmer not in counts:
                counts[kmer] = 0
            counts[kmer] += 1
    return counts


def load_response(path: str) -> tuple[DataFrame, DataFrame]:
    """
    load the responses
    :param path:
    :return: pandas DataFrame of the responses.
    """
    lines: List[str] = open(path, "r").readlines()
    # id_to_rate: Dict[str: float] = {}
    # id_to_x0: Dict[str: float] = {}
    early_onset_dict ={}
    late_onset_dict = {}
    for line in lines[1:]:
        line = line.split()
        id, deg_rate, x0, t0 = line[0], line[1], line[2], line[3]
        # id_to_rate[id],id_to_x0[id] = deg_rate, x0
        if float(t0) == 1:
            early_onset_dict[id] = [deg_rate, x0, t0]
        else:
            late_onset_dict[id] = [deg_rate, x0, t0]

    # rate_df: pd.DataFrame = pd.DataFrame.from_dict(id_to_rate, orient="index")
    # x0_df: pd.DataFrame = pd.DataFrame.from_dict(id_to_x0, orient="index")
    # df: pd.DataFrame = rate_df.join(x0_df, how= "inner")
    early_onset_df: pd.DataFrame = pd.DataFrame.from_dict(early_onset_dict, orient="index")
    late_onset_df: pd.DataFrame = pd.DataFrame.from_dict(late_onset_dict, orient="index")
    early_onset_df.columns,late_onset_df.columns = ['degradation rate', 'x0', 't0'], ['degradation rate', 'x0', 't0']
    return early_onset_df, late_onset_df


