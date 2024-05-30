import matplotlib.pyplot as plt
import pandas as pd

expr_rate = pd.read_csv(
    "./Neural_network/DeepUTR-main/files/5utr_dataset/data.expr.5UTR_40A.csv"
    "", index_col='id')
# expr_rate['6hr'] = (expr_rate['6hr'] + expr_rate['6hr.1']) / 2
expr_rate.drop('6hr.1', axis=1, inplace=True)
# expr_rate.interpolate(method='linear', inplace=True)
# print(expr_rate)
lr_as_true_res = pd.read_csv(
    "./Neural_network/DeepUTR-main/files/5utr_dataset/data.dg.5UTR_40A.csv",
    index_col='id', skiprows=0)
lasso_pred = pd.read_csv("./results/prediction_of_seqs_with_motifs.csv",

                         index_col=0, header=None)

for seq_id, dg in lasso_pred.iterrows():
    xes = [0, 1, 3, 5, 6, 7, 8]

    plt.plot(xes[1:], [expr_rate.loc[seq_id][f"{x}hr"] for x in xes[1:]],
             marker='o', linestyle='none')

    x0 = lr_as_true_res.loc[seq_id]['x0']

    lasso_pred_func = lambda x: (-dg[1] * x) + x0
    lasso_predictions = [lasso_pred_func(x) for x in xes]
    plt.plot(xes, lasso_predictions, label="Lasso prediction")

    lr_dg = lr_as_true_res.loc[seq_id]['dg']
    lr_func = lambda x: -lr_dg * x + x0
    lr_predictions = [lr_func(x) for x in xes]
    plt.plot(xes, lr_predictions, label="Linear Regression")

    plt.title(seq_id)
    plt.legend()
    plt.savefig(f"./results/predVsTrue/{seq_id}.png")
    plt.clf()
