import itertools
from typing import IO, Dict

import numpy as np
import pandas as pd


def load_seq_data(path: str, min_length: int, max_length: int) -> pd.DataFrame:
    """
    loads the data from a text file into data frame that count how many times
    each sequence in the length of 3-7 nucleotides is in each 3'URR.
    :param path: path to a text file
    :return: data frame that count how many times each sequence in the length of
     3-7 nucleotides is in each 3'URR. The shape (n_genes, 4^3+4^4+4^5+4^6+4^7)
    """
    f: list = open(path, "r").readlines()

    # make all the possible sequence in the length of 3-7 nt.
    characters = 'augc'

    all_sequences = []

    for length in range(min_length, max_length + 1):
        sequences_of_length = [''.join(p) for p in
                               itertools.product(characters, repeat=length)]
        all_sequences.extend(sequences_of_length)

    df: pd.DataFrame = pd.DataFrame(columns=all_sequences)

    k_mers_counter: Dict[str: Dict[str, int]] = {}
    for line in f[2:]:
        line = line.split()
        id, seq = line[0], line[1]
        k_mers_counter[id] = k_mers_count(seq, min_length, max_length)

    # append row for each gene to the data frame
    for id in k_mers_counter.keys():
        df.loc[id] = np.zeros(df.shape[1], dtype=int)
        for k_mer in k_mers_counter[id].keys():
            df.at[id, k_mer] = k_mers_counter[id][k_mer]
    pass


def k_mers_count(seq: str, min_length: int, max_length: int):
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
            # Add the kmer to the dictionary if it's not there
            if kmer not in counts:
                counts[kmer] = 0
            # Increment the count for this kmer
            counts[kmer] += 1
    return counts