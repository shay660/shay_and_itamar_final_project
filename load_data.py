from typing import Dict, List, Tuple
import pandas as pd

def matrix_generator(f : pd.DataFrame, response_vec, min_length: int,
                     max_length: int) -> pd.DataFrame:
    """
    generate a matrix from a list of sequences that count how many times each
    gene of the length min_length to max_length is in each sequence.
    :param seq: list of sequences
    :param max_length:the min length of k_mer to look for.
    :param min_length:the max length of k_mer to look for.
    :param response_vec:
    :return: data frame that count how many times each sequence in the length of
     3-7 nucleotides is in each 3'URR. The shape (n_genes, 4^3+4^4+4^5+4^6+4^7)
    """

    # create dict. key is id, value is a dict where the key is k_mer and the value is its frequency in the id seq.
    k_mers_counter: Dict[str: Dict[str, int]] = {}
    for sample in f.index:
        #  checks if we have the response value for the id before loading it to the dataset
        if sample in response_vec.index.tolist():
            k_mers_counter[sample] = k_mers_count(f.loc[sample, 'seq'], min_length, max_length)

    samples = pd.DataFrame.from_dict(k_mers_counter, orient="index").join(response_vec, how="inner")
    samples.fillna(0, inplace=True)
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


