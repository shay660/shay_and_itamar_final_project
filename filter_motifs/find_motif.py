# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import re

TTTA_MOTIF_REGEX = r"TTTTTTTTTTT*A"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    all_sequences = pd.read_csv("../Neural_network/DeepUTR-main/files/5utr_dataset/mRNA_sequences_for_5utr_0A.csv")
    motif_sequences = all_sequences[all_sequences['seq'].str.contains(TTTA_MOTIF_REGEX)]
    motif_sequences.head()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
