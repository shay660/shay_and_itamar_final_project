import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
# from linear_regression.plot_weights import plot_weights

relevant_ids = [
    "S1_L_T23640",
    "S3_H_T21650",
    "S3_H_T14986",
    "S1_L_T15944",
    "S3_H_T17074",
    "S1_L_T47494",
    "S3_H_T4914",
    "S1_L_T37138",
    "S2_H_T2059",
    "S1_L_T23724",
    "S1_L_T44541",
    "S3_H_T8130"
]


def plot_true_vs_prd_graphs():
    expr_rate, lasso_pred, lr_as_true_res = open_files()
    max_y = max(lr_as_true_res.loc[lasso_pred.index, :]['x0']) + .5
    min_y = min(expr_rate.loc[lasso_pred.index, :]['8hr']) * 0.5
    for seq_id, dg in lasso_pred.iterrows():
        xes = [0, 1, 3, 5, 6, 7, 8]

        x0 = lr_as_true_res.loc[seq_id]['x0']
        y = [x0] + [expr_rate.loc[seq_id][f"{x}hr"] for x in xes[1:]]

        plt.plot(xes
                 , y,
                 marker='o', linestyle='none')
        pred_x0 = dg['Predicted_x0']
        pred_slope = -dg.loc['Predicted_Degradation_Rate']
        lasso_pred_func = lambda x: (pred_slope*x) + x0
        lasso_predictions = [lasso_pred_func(x) for x in xes]
        plt.plot(xes, lasso_predictions, label="Lasso prediction")

        # lr_dg = lr_as_true_res.loc[seq_id]['dg']
        # lr_func = lambda x: -lr_dg * x + x0
        # lr_predictions = [lr_func(x) for x in xes]
        coef = np.polyfit(xes, y, 1)
        polynom = np.poly1d(coef)
        trandline_y = polynom(xes)
        plt.plot(xes, trandline_y, label="Linear Regression")
        plt.ylim(min_y, max_y)
        half_life = np.log(2) / -coef[0]
        plt.text(.95, .85, f"Half-life = {half_life:.3f}",
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right')

        # plt.title(seq_id)
        plt.legend()
        plt.savefig(f"./results/predVsTrue/new_seqs/{seq_id}.png", dpi=150)
        # plt.figure(figsize=(12, 6))
        plt.clf()


def open_files():
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
    lasso_pred = pd.read_csv(
        "models/5utr_40A_2024-05-07_16-27-53/linear_regresseion_5utr_40A_results.csv",
        index_col=0)
    lasso_pred.drop_duplicates(inplace=True)
    lasso_pred = lasso_pred.loc[relevant_ids]
    return expr_rate, lasso_pred, lr_as_true_res


def merge_plots(input_path, output_path):
    kmer_plot_path = "results/5utr_k_mers_plots"
    kmers_plots = os.listdir(kmer_plot_path)
    pred_vs_true_path = input_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for plot in os.listdir(pred_vs_true_path):
        image1 = Image.open(f"{pred_vs_true_path}/{plot}")
        plot2 = None
        for p in kmers_plots:
            if plot.split('.')[0] in p:
                plot2 = p
                break
        try:
            image2 = Image.open(f"{kmer_plot_path}/{plot2}")
        except AttributeError:
            print(f"sequence {plot} not found")
            continue

        width1, height1 = image1.size
        width2, height2 = image2.size
        combine_width = max(width1, width2)
        combine_height = height1 + height2

        combine_image = Image.new("RGB", (combine_width, combine_height))
        combine_image.paste(image2, (0, 0))
        combine_image.paste(image1, (0, height2))
        combine_image.save(f"{output_path}{plot}")


if __name__ == '__main__':
    path_to_weights = "models/5utr_40A_2024-05-07_16-27-53/most_significant_kmers.csv"
    # plot_weights(path_to_weights)
    plot_true_vs_prd_graphs()
    merge_plots("results/predVsTrue/new_seqs/",
                "results/combine_plots/new_seqs/")
