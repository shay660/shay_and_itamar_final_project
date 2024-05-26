import pandas as pd

raw_data = pd.read_csv("data_for_linear_reg\data.dg.5UTR_40A_excel.csv", index_col = 'id')
raw_sequences = pd.read_csv("data_for_linear_reg\mRNA_sequences_for_5utr_40A.csv", index_col = 'id')
mse_factor, rsq_factor = raw_data['mse'].quantile(0.25), raw_data['rsq'].quantile(0.75)
high_cor_data = raw_data[(raw_data['mse'] < mse_factor)|(raw_data['rsq'] > rsq_factor)]
train_set_for_lasso = pd.merge(high_cor_data, raw_sequences, left_index=True, right_index=True, how='inner')
train_set_for_lasso['seq'].to_csv("train_seq_lasso.csv",index= True)
train_set_for_lasso[['dg', 'x0', 't0']].to_csv("train_labels_lasso.csv",index= True)
pass