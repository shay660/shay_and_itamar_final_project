import itertools
from typing import IO, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_seq_data(samples_path: str, response_path: str, min_length: int,
                  max_length: int) -> pd.DataFrame:
    """
    loads the data from a text file into data frame that count how many times
    each sequence in the length of 3-7 nucleotides is in each 3'URR.
    :param path: path to a text file
    :return: data frame that count how many times each sequence in the length of
     3-7 nucleotides is in each 3'URR. The shape (n_genes, 4^3+4^4+4^5+4^6+4^7)
    """
    f: list = open(samples_path, "r").readlines()

    # make all the possible sequence in the length of 3-7 nt.
    # characters = 'ATGC'
    # all_sequences = []
    # for length in range(min_length, max_length + 1):
    #     sequences_of_length = [''.join(p) for p in
    #                            itertools.product(characters, repeat=length)]
    #     all_sequences.extend(sequences_of_length)
    # loads the responses vector
    response_vec = load_response(response_path)
    # create dict. key is id, value is a dict where the key is k_mer and the
    # value is it's frequency in the id seq.
    k_mers_counter: Dict[str: Dict[str, int]] = {}
    for line in tqdm(f[2:]):
        line = line.split()
        id, seq = line[0], line[1]
        #checks if we have the response value for the id before loading it to the dataset
        if id in response_vec.index.tolist() and float(response_vec.loc[id, 't0']) == 1:
            k_mers_counter[id] = k_mers_count(seq, min_length, max_length)

    samples = pd.DataFrame.from_dict(k_mers_counter, orient="index", dtype=int)
    samples.fillna(0, inplace=True)
    df: pd.DataFrame = samples.join(response_vec, how="inner")
    return df


def k_mers_count(seq: str, min_length: int, max_length: int) -> Dict[str, int]:
    """

    :param seq:
    :param k:
    :return:
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


def load_response(path: str) -> pd.DataFrame:
    """
    load the respon
    :param path:
    :return:
    """
    lines: List[str] = open(path, "r").readlines()
    # id_to_rate: Dict[str: float] = {}
    # id_to_x0: Dict[str: float] = {}
    id_dict ={}
    for line in lines[1:]:
        line = line.split()
        print(line)
        id, deg_rate, x0, t0 = line[0], line[1], line[2], line[3]
        # id_to_rate[id],id_to_x0[id] = deg_rate, x0
        id_dict[id] = [deg_rate, x0, t0]


    # rate_df: pd.DataFrame = pd.DataFrame.from_dict(id_to_rate, orient="index")
    # x0_df: pd.DataFrame = pd.DataFrame.from_dict(id_to_x0, orient="index")
    # df: pd.DataFrame = rate_df.join(x0_df, how= "inner")
    df: pd.DataFrame = pd.DataFrame.from_dict(id_dict, orient="index")
    df.columns = ['degradation rate', 'x0', 't0']
    return df
