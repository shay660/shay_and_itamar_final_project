import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def main():
    expr_rate, lasso_pred, lr_as_true_res = open_files()
    y_max = max(lr_as_true_res.loc[lasso_pred.index]['x0']) + 0.5
    y_min = min(expr_rate.loc[lasso_pred.index]['8hr'])
    plot_graphs(expr_rate, lasso_pred, lr_as_true_res, y_max, y_min)


def plot_graphs(expr_rate, lasso_pred, lr_as_true_res, y_max, y_min):
    for seq_id, dg in lasso_pred.iterrows():
        xes = [0, 1, 3, 5, 6, 7, 8]

        x0 = lr_as_true_res.loc[seq_id]['x0']
        y = [x0] + [expr_rate.loc[seq_id][f"{x}hr"] for x in xes[1:]]

        plt.plot(xes, y, marker='o', linestyle='none')
        lasso_pred_func = lambda x: (dg.loc['pred_dg'] * x) + x0
        lasso_predictions = [lasso_pred_func(x) for x in xes]
        plt.plot(xes, lasso_predictions, label="Lasso prediction")

        coef = np.polyfit(xes, y, 1)
        polynom = np.poly1d(coef)
        trandline_y = polynom(xes)
        plt.plot(xes, trandline_y, label=f"Linear Regression")
        plt.ylim(y_min, y_max)
        half_life = np.log(2) / -coef[0]
        plt.axvline(x=half_life, color='r', linestyle='none',
                    label=f'Half-life: {half_life:.3f} hours')

        plt.xlabel("Time (hours)")
        plt.ylabel("expression label ")
        plt.title(seq_id)
        plt.legend()
        plt.savefig(f"./results/predVsTrue2/{seq_id}.png")
        plt.figure(figsize=(12, 8))
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
    lasso_pred = pd.read_csv("./data/final_seqs_with_pred.csv",
                             index_col='id')
    lasso_pred.drop_duplicates(inplace=True)
    return expr_rate, lasso_pred, lr_as_true_res


def merge_plots():
    # Open the images
    image1 = Image.open('./results/predVsTrue2/S1_L_T14419.png')
    image2 = Image.open('./results/predVsTrue2/S1_H_T4422.png')

    # Get the dimensions of the images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Calculate the total width and height for the combined image
    total_width = width1 + width2
    max_height = max(height1, height2)

    # Create a new image with the combined width and max height
    combined_image = Image.new('RGBA', (total_width, max_height),
                               (255, 255, 255, 0))

    # Paste the first image onto the combined image
    combined_image.paste(image1, (0, 0))

    # Paste the second image beside the first one
    combined_image.paste(image2, (width1, 0))

    # Save the combined image
    combined_image.save('combined_image.png')

if __name__ == '__main__':
    # main()
    merge_plots()
