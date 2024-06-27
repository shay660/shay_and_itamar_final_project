import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# from linear_regression.plot_weights import plot_weights

relevant_ids = [
    "S1_L_T5728", "S1_L_T17256", "S3_H_T14986", "S2_L_T6744", "S3_H_T8300"
]


def plot_true_vs_prd_graphs(expr_path, dg_file_path, path_to_prediction,
                            output_path):
    expr_rate, lasso_pred, lr_as_true_res = open_files(expr_path=expr_path,
                                                       dg_file_path=dg_file_path,
                                                       path_to_predictions=path_to_prediction)
    max_y = max(lr_as_true_res.loc[lasso_pred.index, :]['x0']) * 1.25
    min_y = min(expr_rate.loc[lasso_pred.index, :]['8hr']) * 0.5
    new_lr_vs_old_lr = {"Old Linear Regression": [],
                        "Old Linear Regression x0": [],
                        "Our Linear Regression": [],
                        "Our Linear Regression x0": []}
    xes = [0, 1, 3, 5, 6, 7, 8]
    for seq_id, dg in lasso_pred.iterrows():

        # x0 = lr_as_true_res.loc[seq_id]['x0']
        y = [expr_rate.loc[seq_id][f"{x}hr"] for x in xes[1:]]
        plt.plot(xes[1:], y, marker='o', linestyle='none')
        # pred_x0 = dg['Predicted_x0']

        lr_dg = lr_as_true_res.loc[seq_id]['dg']
        # lr_func = lambda x: -lr_dg * x + x0
        # lr_predictions = [lr_func(x) for x in xes]
        coef = np.polyfit(xes[1:], y, 1)
        polynom = np.poly1d(coef)
        trandline_y: np.ndarray = polynom(xes)
        # trandline_y = np.concatenate(([x0], trandline_y[1:]))
        plt.plot(xes, trandline_y, label="Linear Regression")
        plt.ylim(min_y, max_y)

        pred_slope = -dg.loc['pred']
        lasso_pred_func = lambda x: (pred_slope * x) + coef[1]
        lasso_predictions = [lasso_pred_func(x) for x in xes]
        plt.plot(xes, lasso_predictions, label="CNN prediction")
        # half_life = np.log(2) / -coef[0]
        mse = dg['MSE']
        plt.text(.95, .85, f"mse= {mse:.6f}",
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right')

        plt.title(seq_id)
        plt.legend()
        plt.savefig(f"{output_path}{seq_id}.png", dpi=300)
        plt.figure(figsize=(12, 6))
        plt.clf()
    pd.DataFrame(new_lr_vs_old_lr).to_csv(output_path +
                                          "/Different_Between_lr.csv",
                                          index=False)


def open_files(expr_path, dg_file_path, path_to_predictions):
    expr_rate = pd.read_csv(expr_path, index_col='id')
    # expr_rate['6hr'] = (expr_rate['6hr'] + expr_rate['6hr.1']) / 2
    expr_rate.drop('6hr.1', axis=1, inplace=True)
    # expr_rate.interpolate(method='linear', inplace=True)
    # print(expr_rate)
    lr_as_true_res = pd.read_csv(dg_file_path,
                                 index_col='id', skiprows=0)
    try:
        lasso_pred = pd.read_csv(path_to_predictions, index_col='Id')
    except ValueError:
        lasso_pred = pd.read_csv(path_to_predictions, index_col=0)

    lasso_pred.drop_duplicates(inplace=True)
    # lasso_pred = lasso_pred.loc[relevant_ids]
    return expr_rate, lasso_pred, lr_as_true_res


def merge_plots(input_path, kmer_plot_path, output_path):
    kmers_plots = os.listdir(kmer_plot_path)
    pred_vs_true_path = input_path
    for plot in os.listdir(pred_vs_true_path):
        image1 = Image.open(f"{pred_vs_true_path}{plot}")
        plot2 = None
        for p in kmers_plots:
            if plot.split('Vs')[0] in p:
                plot2 = p
                break
        try:
            image2 = Image.open(f"{kmer_plot_path}/{plot2}")
            image2 = image2.resize((int(image1.size[0] * 1.5), image1.size[1]))
        except AttributeError:
            print(f"sequence {plot} not found")
            continue

        width1, height1 = image1.size
        width2, height2 = image2.size
        combine_width = max(width1, width2)
        combine_height = height1 + height2
        # combine_height = max(height1, height2)
        # combine_width = width1 + width2

        combine_image = Image.new("RGB", (combine_width, combine_height))
        combine_image.paste(image2, (0, 0))
        combine_image.paste(image1, (0, height1))
        combine_image.save(f"{output_path}{plot}")


# def plot_old_vs_new(new_seqs_path, old_seqs_path, output_path, is_nn=1):
#     new_seqs = pd.read_csv(new_seqs_path, index_col='id')
#     old_seqs: pd. DataFrame = pd.read_csv(old_seqs_path, index_col='id')
#     old_seqs = old_seqs.loc[relevant_ids]
#     # old_seqs['6hr'] = (old_seqs['6hr'] + old_seqs['6hr.1']) / 2
#     old_seqs.drop('6hr.1', axis=1, inplace=True)
#     # y_max = max(old_seqs['x0']) * 1.2
#     xes = [0, 1, 3, 5, 6, 7, 8]
#     for seq, deg in old_seqs.iterrows():
#         coef = np.polyfit(xes[1:], deg, 1)
#         polynom = np.poly1d(coef)
#         trandline_y = polynom(xes)
#         plt.plot(xes, trandline_y, label=f"Real Data (HL: "
#                                          f"{round(np.log(2) / -coef[0], 3)})")
#         new_seq_name = None
#         for new_seq, new_dg in new_seqs.iterrows():
#             if seq in new_seq:
#                 new_seq_name = new_seq
#                 # pred_x0 = new_dg.loc['x0']
#                 pred_slope = is_nn * new_dg.loc['deg rate']
#                 prediction_function = lambda x: coef[1] + (pred_slope * x)
#                 plt.plot(xes, [prediction_function(x) for x in xes],
#                          label=f" {new_seq} "
#                                f"(HL: {round(np.log(2) / -pred_slope, 3)})")
#
#         plt.ylim(3, 13)
#         plt.title(seq)
#         plt.legend()
#         plt.savefig(f"{output_path}{new_seq_name}Vs{seq}.png")
#         # plt.show()
#         plt.clf()

def plot_old_vs_new(new_seqs_path, old_seqs_path, output_path, is_nn=1):
    new_seqs = pd.read_csv(new_seqs_path, index_col='id')
    old_seqs: pd.DataFrame = pd.read_csv(old_seqs_path, index_col='id')
    old_seqs = old_seqs.loc[relevant_ids]
    # old_seqs['6hr'] = (old_seqs['6hr'] + old_seqs['6hr.1']) / 2
    old_seqs.drop('6hr.1', axis=1, inplace=True)
    # y_max = max(old_seqs['x0']) * 1.2
    xes = [0, 1, 3, 5, 6, 7, 8]
    mock_x0 = 11
    for new_seq, prediction in new_seqs.iterrows():
        # coef = np.polyfit(xes[1:], deg, 1)
        # polynom = np.poly1d(coef)
        # trandline_y = polynom(xes)

        do_not_plot: bool = False
        for old_seq, old_deg in old_seqs.iterrows():
            if old_seq in new_seq:
                if old_seq == new_seq:
                    do_not_plot = True
                    break
                coef = np.polyfit(xes[1:], old_deg, 1)
                polynom = np.poly1d(coef)
                # trandline_y = polynom(xes)
                plt.plot(xes, [x * coef[0] + mock_x0 for x in xes], label=f""
                                                                          f" {old_seq} "
                                                                          f"(HL: {round(np.log(2) / coef[0], 3)})")
        if do_not_plot:
            plt.clf()
            continue

        pred_slope = is_nn * prediction.loc['deg rate']
        # prediction_function = lambda x: coef[1] + (pred_slope * x)
        predict_x0 = prediction.loc['x0']
        prediction = [(pred_slope * x) + mock_x0 for x in xes]
        plt.plot(xes, prediction, label=f"{new_seq} (HL: "
                                        f"{round(np.log(2) / pred_slope, 3)})")
        plt.ylim(3, 13)
        plt.title(new_seq)
        plt.legend()
        plt.savefig(f"{output_path}{new_seq}.png")
        # plt.show()
        plt.clf()


def compare_lr(expr_data_path, lr_path, deg_or_dg, output_path):
    expr_data = pd.read_csv(expr_data_path, index_col='id')
    filter_seq = pd.read_csv("data/final_seqs_with_pred.csv", index_col='id')
    expr_data = expr_data.loc[filter_seq.index]
    expr_data['6hr'] = (expr_data['6hr'] + expr_data['6hr.1']) / 2
    expr_data.drop('6hr.1', axis=1, inplace=True)

    old_lr = pd.read_csv(lr_path, index_col='id')
    xes = [1, 3, 5, 6, 7, 8]
    # new_lr_vs_old_lr = {"seq": [],
    #                     "Old Linear Regression": [],
    #                     "Old Linear Regression x0": [],
    #                     "Our Linear Regression": [],
    #                     "Our Linear Regression x0": []}
    i = 0
    for seq_id, dg in expr_data.iterrows():
        old_x0 = old_lr.loc[seq_id]['x0']
        old_dg = old_lr.loc[seq_id][deg_or_dg]

        coef = np.polyfit(xes, dg, 1)
        # new_lr_vs_old_lr["seq"].append(seq_id)
        # new_lr_vs_old_lr["Old Linear Regression"].append(old_dg)
        # new_lr_vs_old_lr["Our Linear Regression"].append(-coef[0])
        # new_lr_vs_old_lr["Old Linear Regression x0"].append(old_x0)
        # new_lr_vs_old_lr["Our Linear Regression x0"].append(coef[1])
        if i < 10:
            polynom = np.poly1d(coef)
            trandline_y = polynom(xes)
            our_hl = round(np.log(2) / -coef[0], 3)
            plt.plot(xes, trandline_y, label=f"Our Linear Regression, "
                                             f"HL: {our_hl}")

            # lr_dg = lr_as_true_res.loc[seq_id]['dg']
            lr_func = lambda x: -old_dg * x + old_x0
            lr_predictions = [lr_func(x) for x in xes]
            y = [dg[f"{x}hr"] for x in xes]
            old_hl = round(np.log(2) / old_dg, 3)
            plt.plot(xes, lr_predictions, label=f"Former Linear Regression, " \
                                                f"HL: {old_hl}")
            plt.plot(xes, y, marker='o', linestyle='none')
            ratio = ((old_hl - our_hl) / old_dg) * 100
            plt.annotate(f'{ratio:.2f}%', xy=(xes[-1], trandline_y[-1]),
                         xytext=(5, 5),
                         textcoords='offset points', fontsize=8, color='blue')
            plt.title(seq_id)
            plt.legend()
            # plt.text(.95, .85, f"Difference as  = {half_life:.3f}",
            #          transform=plt.gca().transAxes, fontsize=12,
            #          verticalalignment='top', horizontalalignment='right')
            plt.savefig(f"{output_path}/{seq_id}.png")
            plt.clf()
            # i += 1
    # pd.DataFrame(new_lr_vs_old_lr).to_csv(
    #     f"{output_path}/old_lr_vs_our_lr.csv",
    #     index=False)


if __name__ == '__main__':
    output_path = "./results/predVsTrue/NN_test_set/"
    path_to_weights = "data/final_seqs_with_pred.csv"
    # plot_weights(path_to_weights)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    plot_true_vs_prd_graphs(expr_path="Neural_network/DeepUTR-main/files"
                                      "/5utr_dataset/data.expr.5UTR_40A.csv",
                            dg_file_path="Neural_network/DeepUTR-main/files"
                                         "/5utr_dataset/data.dg.5UTR_40A.csv",
                          path_to_prediction="Neural_network/DeepUTR-main"
                                             "/files/results/new_seqs_predictions/test_set_CNN+_rate_with_ids.csv",
                            output_path=output_path)
    # plot_old_vs_new(
    #     new_seqs_path="data/final_seqs_40A_model.csv",
    #     old_seqs_path="./Neural_network/DeepUTR-main/files"
    #                   "/5utr_dataset/data.expr.5UTR_40A.csv",
    #     output_path="results/predVsTrue/final_seqs/new_final_seqs/without_x0/",
    #     is_nn=-1)
    # merge_plots("results/predVsTrue/final_seqs/",
    #             "results/5utr_k_mers_plots/extra_new_seqs/",
    #             "results/combine_plots/new_final_seqs/")

    # compare_lr("./Neural_network/DeepUTR-main/files/5utr_dataset/data.expr"
    #            ".5UTR_40A.csv",
    #            "./Neural_network/DeepUTR-main/files/5utr_dataset/data.dg"
    #            ".5UTR_40A.csv", "dg",
    #            "linear_regression/compare_lr")
