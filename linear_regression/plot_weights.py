import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Load the data
WEIGHTS_PATH = "data_for_linear_reg/weights_data/weights_vector.csv"
SEQUENCE_PATH = "data_for_linear_reg/final_all.csv"


sequences = pd.read_csv(SEQUENCE_PATH, index_col=0, header=0).drop_duplicates()

no_lasso_model = joblib.load('../linear_regression/data_for_linear_reg/models/5utr_no_lasso_model.joblib')
lasso_model = joblib.load("../linear_regression/data_for_linear_reg/models/filtered_data_5utr_40A_model.joblib")

no_lasso_weights = no_lasso_model.coef_[0, :]
lasso_weights = lasso_model.coef_[0, :]

no_lasso_k_mer_names = no_lasso_model.feature_names_in_
lasso_k_mer_names = lasso_model.feature_names_in_

weights = pd.DataFrame({'k_mer': no_lasso_k_mer_names, 'weight': no_lasso_weights}).set_index('k_mer')
lasso_weights_df = pd.DataFrame({'k_mer': lasso_k_mer_names, 'weight': lasso_weights}).set_index('k_mer')


# Define a function to plot sequence weights
def plot_sequence_weights(seq_weights, seq_arr, title, filepath, max_val):
    plt.figure(figsize=(20, 6))
    sns.set(style="whitegrid")
    sns.lineplot(x=seq_weights.keys(), y=seq_weights.values(), marker='o', color='b', linewidth=2.5, markersize=8)

    plt.xticks(range(0, len(seq_arr)), seq_arr, fontsize=10)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Signal strength estimation', fontsize=14)
    plt.ylim(-max_val*1.1,max_val*1.1)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


# Iterate over sequences for lasso_weights
plots_data= {}
max_val = 0
for sample in sequences.index:
    sequence = sequences.loc[sample]['seq']
    seq_weights = {}
    for pos in range(len(sequence)):

        pos_weight = 0
        for k_len in range(3, 8):
            if pos+k_len< len(sequence):
                k_mer = sequence[pos: pos + k_len]
                if k_mer in lasso_weights_df.index:
                    pos_weight += float(lasso_weights_df.loc[k_mer])
        seq_weights[pos] = pos_weight
    seq_arr = list(sequence)
    preprocessed_weights = list(seq_weights.values())
    for pos in seq_weights.keys():
        for neighbor in [-2,-1,1,2]:
            if pos + neighbor >= 0 and pos + neighbor < len(sequence):
                seq_weights[pos] += (preprocessed_weights[pos+neighbor])/((abs(neighbor)+1)**2)
        seq_weights[pos] = (1/(1 + np.exp(-seq_weights[pos])) - 0.5)**3
    plots_data[sample] = seq_weights
    if max_val < max(abs(min(seq_weights.values())),max(seq_weights.values())):
        max_val = max(abs(min(seq_weights.values())),max(seq_weights.values()))
for sample in plots_data.keys():
    plot_sequence_weights(plots_data[sample], list(sequences.loc[sample]['seq']), f"{sample} - Lasso",
                      f'results/5utr_k_mer_plots_lasso/sequence_{sample}_lasso.png', max_val)
