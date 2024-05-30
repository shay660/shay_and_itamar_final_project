import joblib
import pandas as pd
import matplotlib.pyplot as plt

WEIGHTS_PATH = "data_for_linear_reg/weights_data/weights_vector.csv"
SEQUENCE_PATH = "data_for_linear_reg/mRNA_sequences_for_5utr_40A.csv"
# weights= pd.read_csv(WEIGHTS_PATH, index_col= 'Feature')

sequences = pd.read_csv(SEQUENCE_PATH)

model = joblib.load('../models/all_data_early_onset_without_polyA_2023-09-27_11-43-19/all_data_early_onset_without_polyA_model.joblib')
weighter = model.coef_[0, :]
k_mer_names= model.feature_names_in_
weights = pd.DataFrame({'k_mer': k_mer_names, 'weight': weighter}).set_index('k_mer')

pass

for index, sequence in sequences['seq'].items():  # Iterate over sequences along with their index
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

    # seq_arr = [f"{char}\n{pos}" for char, pos in zip(sequence, range(1, len(sequence) + 1))]
    seq_arr = list(sequence)
    # Plotting
    plt.plot(seq_weights['pos'], seq_weights['weight'], marker='o')

    # Set x-axis ticks and labels
    plt.xticks(range(1, len(sequence) + 1), seq_arr)

    # Save and show plot
    plt.savefig(f'results/3utr_k_mer_plots/sequence_{index + 1}.png')
    plt.show()
    plt.clf()


