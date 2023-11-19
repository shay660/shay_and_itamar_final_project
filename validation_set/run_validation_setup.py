import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from validation_set_setup import find_best_parameter, make_plot



def main():
    df = pd.read_csv("normalized_mRNA_levels_Aplus.csv")
    X = np.array(range(1, 10)).reshape(-1, 1)
    deg_rates = np.array([], dtype=float)
    x0_list = np.array([], dtype=float)

    for i, row in df.iterrows():
        seq_id = row[0]
        t0, models_lst, y_pred = find_best_parameter(X, row[1:])

        # Plot the data and the selected model's predictions
        make_plot(X, row[1:], y_pred, seq_id)
        # print("Step Location step_loc:", t0)

        # find slope
        best_model: LinearRegression = models_lst[t0][bool(t0)]
        slope = best_model.coef_[0]
        deg_rates = np.append(deg_rates, slope)
        x0 = models_lst[t0][0].intercept_
        x0_list = np.append(x0_list, x0)

    # add to the df
    df['degradation rate'] = deg_rates
    df['x0'] = x0_list
    # df.to_csv("normalized_mRNA_levels_Aplus_with_deg_rate.csv")


if __name__ == '__main__':
    main()


