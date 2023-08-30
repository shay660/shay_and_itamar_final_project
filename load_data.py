from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm


def load_seq_data(samples_path: str, responses_vec: pd.DataFrame, min_length:
 int, max_length: int) -> pd.DataFrame:  # pd.DataFrame, pd.DataFrame
    """
    loads the data from a text file into data frame that count how many times
    each sequence in the length of 3-7 nucleotides is in each 3'URR.
    :param samples_path: path to a text file
    :param responses_vec:
    :param late_onset_responses:
    :param min_length:
    :param max_length:
    :return: Two data frame that count how many times each sequence in the
    length of 3-7 nucleotides is in each 3'URR. one for the early onset and
    the other for late onset sequences. The shape (n_genes, 4^3+4^4+4^5+4^6+4^7)
    """

    # loads the responses vector
    # responses_vec, late_onset_responses = load_response(
    #     response_path)

    df = _matrix_generator(f, max_length, min_length,
                           responses_vec)
    return df


def _matrix_generator(f: List[str], response_vec,min_length: int,
                      max_length:int) -> pd.DataFrame:
    """
    generate a matrix from a list of sequences that count how many times each
    gene of the length min_length to max_length is in each sequence.
    :param f: list of sequences
    :param max_length:the min length of k_mer to look for.
    :param min_length:the max length of k_mer to look for.
    :param response_vec:
    :return: data frame that count how many times each sequence in the length of
     3-7 nucleotides is in each 3'URR. The shape (n_genes, 4^3+4^4+4^5+4^6+4^7)
    """
    # create dict. key is id, value is a dict where the key is k_mer and the
    # value is its frequency in the id seq.
    k_mers_counter: Dict[str: Dict[str, int]] = {}
    for line in tqdm(f[2:]):
        line = line.split()
        id, seq = line[0], line[1]
        #  checks if we have the response value for the id before loading it
        #  to the dataset
        if id in response_vec.index.tolist():
            k_mers_counter[id] = k_mers_count(seq, min_length, max_length)
    samples = pd.DataFrame.from_dict(k_mers_counter, orient="index", dtype=int)
    samples.fillna(0, inplace=True)
    # samples: pd.DataFrame = samples.join(response_vec, how="inner")
    return samples


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


def load_response(path: str, ) -> DataFrame:
    """
    load the responses
    :param path:path to the responses
    :return: pandas DataFrame of the responses.
    """
    lines: List[str] = open(path, "r").readlines()
    # id_to_rate: Dict[str: float] = {}
    # id_to_x0: Dict[str: float] = {}
    all_onset_dict = {}
    for line in lines[1:]:
        line = line.split()
        id, deg_rate, x0, t0 = line[0], line[1], line[2], line[3]
        # id_to_rate[id],id_to_x0[id] = deg_rate, x0
        all_onset_dict[id] = [deg_rate, x0, t0]

    # rate_df: pd.DataFrame = pd.DataFrame.from_dict(id_to_rate, orient="index")
    # x0_df: pd.DataFrame = pd.DataFrame.from_dict(id_to_x0, orient="index")
    # samples: pd.DataFrame = rate_df.join(x0_df, how= "inner")
    all_onset_df: pd.DataFrame = pd.DataFrame.from_dict(all_onset_dict,
                                                        orient="index")
    all_onset_df.columns = ['degradation rate', 'x0', 't0']
    return all_onset_df
