import pandas as pd
import numpy as np
import torch
# import networkx as nx
from sklearn.linear_model import LinearRegression
from RNA_dataset import RNADataset


class Dataprocessor:
    def __init__(self, sequences_file=None, deg_rates_file=None,
                 split_ids_file=None):
        sequences_df: pd.DataFrame = pd.read_csv(sequences_file, index_col=0)
        deg_rates_df: pd.DataFrame = pd.read_csv(deg_rates_file, index_col=0)

        sequence_tensor = self.generate_one_hot_encoding_matrix(
            sequences_df.values)

        # Compute degradation rates for all sequences
        deg_rates = self.compute_LR_slopes(deg_rates_df.values)
        deg_rates_tensor = torch.tensor(deg_rates, dtype=torch.float32)

        self.all_data = RNADataset(sequence_tensor, deg_rates_tensor)

        if split_ids_file:
            indices_df = pd.read_csv(split_ids_file)

            self.train_set = self._prepare_data(indices_df['train_ids'],
                                                sequence_tensor,
                                                deg_rates_tensor)
            self.validation_set = self._prepare_data(
                indices_df['validation_ids'].dropna(), sequence_tensor,
                deg_rates_tensor)
            self.test_set = self._prepare_data(indices_df['test_ids'].dropna(),
                                               sequence_tensor,
                                               deg_rates_tensor)

        print("Finished processing the data")

    @staticmethod
    def _prepare_data(indices, sequence_tensor, deg_rates_tensor):
        indices = indices.astype(int).values  # Convert indices to integers
        # and to a numpy array
        one_hot_seq = sequence_tensor[indices]
        deg_rates = deg_rates_tensor[indices]
        return RNADataset(one_hot_seq, deg_rates)

    # @staticmethod
    # def split_the_data(sequences_df):
    #     seq_list = sequences_df["seq"].values.tolist()
    #     hash_map = {}
    #     k = 20  # k-mer length
    #     l = 110  # seqence fixed length
    #     G = nx.Graph()
    #     G.add_nodes_from(range(len(seq_list)))
    #
    #     # passing all k-mer in the seqences and create the edges accordingly
    #     for i, seq in enumerate(seq_list):
    #         for j in range(l - k + 1):
    #             kmer = seq[j:j + k]
    #             if kmer in hash_map:
    #                 hash_map[kmer].add(i)
    #                 edges_list = [(i, m) for m in hash_map[kmer]]
    #                 G.add_edges_from(edges_list)
    #             else:
    #                 hash_map[kmer] = {i}
    #
    #     # create the connected components
    #     connected_components_list = list(nx.connected_components(G))
    #     connected_components_list.sort(key=len, reverse=True)
    #
    #     # create the disjoint  train, validation, and test sets
    #     # This may not deliver the same ids as we provided, since we could not reproduce the random pick
    #     train = []
    #     i = 0
    #     while ((len(train) + len(connected_components_list[i])) <= (80 / 100)
    #            * len(seq_list)):
    #         train = train + list(connected_components_list[i])
    #         i += 1
    #
    #     test = []
    #     while ((len(test) + len(connected_components_list[i])) <= 0.5 * (
    #             len(seq_list) - len(train))):
    #         test = test + list(connected_components_list[i])
    #         i += 1
    #
    #     validation = []
    #     for connected_component in connected_components_list[i:]:
    #         validation = validation + list(connected_component)
    #
    #     # save the ids in a csv file
    #     indices_df = [[id] for id in train]
    #     for i in range(len(validation)):
    #         indices_df[i].append(validation[i])
    #     for i in range(len(test)):
    #         indices_df[i].append(test[i])
    #
    #     indices_df = pd.DataFrame(indices_df,
    #                               columns=['train_ids', 'validation_ids',
    #                                        "test_ids"])
    #     return indices_df

    @staticmethod
    def one_hot_encode_sequence(seq):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        seq_len = len(seq)
        one_hot = np.zeros((seq_len, 4), dtype=np.float32)
        for i, nucleotide in enumerate(seq):
            one_hot[i, mapping[nucleotide]] = 1.0
        return one_hot

    def generate_one_hot_encoding_matrix(self, sequences) -> torch.tensor:
        one_hot_sequences = np.array(
            [self.one_hot_encode_sequence(seq[0]) for seq in sequences])
        # Convert to PyTorch tensors
        return torch.tensor(one_hot_sequences, dtype=torch.float32)

    def get_train_set(self) -> RNADataset:
        return self.train_set

    def get_validation_set(self) -> RNADataset:
        return self.validation_set

    def get_test_set(self) -> RNADataset:
        return self.test_set

    def get_all_data(self) -> RNADataset:
        return self.all_data

    def compute_LR_slopes(seld, values_array):
        num_of_samples = len(values_array)
        # compute the slopes
        if (values_array.shape[1] == 9):
            t = np.matrix([1, 2, 3, 4, 5, 6, 7, 8, 10]).T
        elif values_array.shape[1] == 7:
            t = np.matrix([1, 3, 5, 6, 6.5, 7, 8]).T
        else:
            t = np.matrix([2, 3, 4, 5, 6, 7, 8, 10]).T
        slops = np.zeros((num_of_samples,))
        for i in range(num_of_samples):
            mdl = LinearRegression().fit(np.asarray(t), values_array[i, :])  #
            slops[i] = mdl.coef_[0]  # beta-slope

        return slops

