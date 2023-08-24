import itertools
from typing import IO, Dict, List

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
    characters = 'ATGC'
    all_sequences = []
    for length in range(min_length, max_length + 1):
        sequences_of_length = [''.join(p) for p in
                               itertools.product(characters, repeat=length)]
        all_sequences.extend(sequences_of_length)

    # create dict. key is id, value is a dict where the key is k_mer and the
    # value is it's frequency in the id seq.
    k_mers_counter: Dict[str: Dict[str, int]] = {}
    for line in f[2:]:
        line = line.split()
        id, seq = line[0], line[1]
        k_mers_counter[id] = k_mers_count(seq, min_length, max_length,
                                          all_sequences)

    df: pd.DataFrame = pd.DataFrame.from_dict(k_mers_counter, orient="index")
    return df


def k_mers_count(seq: str, min_length: int, max_length: int,
                 all_sequences: List[str]):
    """

    :param seq:
    :param k:
    :return:
    """
    # Start with an empty dictionary
    counts = {seq: 0 for seq in all_sequences}
    for k in range(min_length, max_length + 1):
        # Calculate how many kmers of length k there are
        num_kmers = len(seq) - k + 1
        # Loop over the kmer start positions
        for i in range(num_kmers):
            # Slice the string to get the kmer
            kmer = seq[i:i + k]
            # # Add the kmer to the dictionary if it's not there
            # if kmer not in counts:
            #     counts[kmer] = 0
            # Increment the count for this kmer
            counts[kmer] += 1
    return counts
