import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
def plot_sequence_weights(seq_weights, seq_arr, title, filepath):
    plt.figure(figsize=(20, 6))
    sns.set(style="whitegrid")
    sns.lineplot(x=seq_weights['pos'], y=seq_weights['weight'], marker='o', color='b', linewidth=2.5, markersize=8)

    plt.xticks(range(1, len(seq_arr) + 1), seq_arr, fontsize=10)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Weight', fontsize=14)
    plt.ylim(-0.06, 0.06)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


# Iterate over sequences for no_lasso_weights
for sample in sequences.index:
    sequence = sequences.loc[sample]['seq']
    seq_weights = {'pos': [], 'weight': []}
    for pos in range(len(sequence)):
        seq_weights['pos'].append(pos + 1)
        pos_weight = 0
        for k_len in range(3, 8):
            try:
                k_mer = sequence[pos: pos + k_len]
                if k_mer in weights.index:
                    pos_weight += float(weights.loc[k_mer])
            except:
                continue
        seq_weights['weight'].append(pos_weight)

    seq_arr = list(sequence)
    plot_sequence_weights(seq_weights, seq_arr, f"{sample} - No Lasso",
                          f'results/5utr_k_mer_plots_no_lasso/sequence_{sample}_no_lasso.png')

# Iterate over sequences for lasso_weights
for sample in sequences.index:
    sequence = sequences.loc[sample]['seq']
    seq_weights = {'pos': [], 'weight': []}
    for pos in range(len(sequence)):
        seq_weights['pos'].append(pos + 1)
        pos_weight = 0
        for k_len in range(3, 8):
            try:
                k_mer = sequence[pos: pos + k_len]
                if k_mer in lasso_weights_df.index:
                    pos_weight += float(lasso_weights_df.loc[k_mer])
            except:
                continue
        seq_weights['weight'].append(pos_weight)

    seq_arr = list(sequence)
    plot_sequence_weights(seq_weights, seq_arr, f"{sample} - Lasso",
                          f'results/5utr_k_mer_plots_lasso/sequence_{sample}_lasso.png')
