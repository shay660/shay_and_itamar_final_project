import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from load_data import matrix_generator


# from sklearn.linear_model import LinearRegression

# from validation_set_setup import find_best_parameter, make_plot


# def main():
#     df = pd.read_csv("normalized_mRNA_levels_Aplus.csv")
#     X = np.array(range(1, 10)).reshape(-1, 1)
#     deg_rates = np.array([], dtype=float)
#     x0_list = np.array([], dtype=float)
#
#     for i, row in df.iterrows():
#         seq_id = row[0]
#         t0, models_lst, y_pred = find_best_parameter(X, row[1:])
#
#         # Plot the data and the selected model's predictions
#         make_plot(X, row[1:], y_pred, seq_id)
#         # print("Step Location step_loc:", t0)
#
#         # find slope
#         best_model: LinearRegression = models_lst[t0][bool(t0)]
#         slope = best_model.coef_[0]
#         deg_rates = np.append(deg_rates, slope)
#         x0 = models_lst[t0][0].intercept_
#         x0_list = np.append(x0_list, x0)
#
#     # add to the df
#     df['degradation rate'] = deg_rates
#     df['x0'] = x0_list
#     # df.to_csv("normalized_mRNA_levels_Aplus_with_deg_rate.csv")
def custom_cost_function(params, X, y):
    """Define the custom cost function for the model"""
    # Extract parameters
    constant_func, linear_decline_slope, intersection_point = params

    # Model prediction
    predictions = np.piecewise(X[:, 0], [
        X[:, 0] <= intersection_point,
        X[:, 0] > intersection_point], [
                                   lambda x: constant_func,
                                   lambda x: linear_decline_slope * x +
                                             constant_func])

    # Compute the mean squared error
    cost = np.mean((predictions - y) ** 2)
    return cost


def main():
    X, deg_rates, df, t0, x0 = init_main()
    to_plot: bool = False
    for name, y in df.iterrows():
        constant_func_opt, intersection_point_opt, linear_decline_slope_opt = \
            calculate_optimal_params(X, y)
        if to_plot:
            make_plot(X, y, name, constant_func_opt, intersection_point_opt,
                      linear_decline_slope_opt)
        deg_rates.append(linear_decline_slope_opt)
        x0.append(constant_func_opt)
        t0.append(intersection_point_opt)
    update_dataframe(deg_rates, df, t0, x0)


def init_main():
    df = pd.read_csv("normalized_mRNA_levels_Aplus.csv", index_col='id')
    X = np.array(range(1, 10), dtype=float).reshape(-1, 1)
    deg_rates, x0, t0 = [], [], []
    return X, deg_rates, df, t0, x0


def update_dataframe(deg_rates, df, t0, x0):
    df['deg_rate'] = deg_rates
    df['x0'] = x0
    df['t0'] = t0
    df.to_csv("normalized_mRNA_levels_Aplus_with_deg_rate.csv")


def calculate_optimal_params(X, y):
    constant_func_opt, linear_decline_slope_opt, intersection_point_opt = \
        0, 0, 0
    min_cost = np.inf
    for i in range(6):
        initial_params = np.ndarray(buffer=np.array([np.mean(y[:i]),
                                                     -0.5, i]), shape=(3,))
        result = minimize(custom_cost_function, initial_params,
                          args=(X, y), method="nelder-mead", tol=0.0005)
        if result.fun < min_cost:
            constant_func_opt, linear_decline_slope_opt, \
                intersection_point_opt = result.x
    return constant_func_opt, intersection_point_opt, linear_decline_slope_opt


def make_plot(X, y, name, constant_func_opt, intersection_point_opt,
              linear_decline_slope_opt):
    plt.plot(X, np.piecewise(X[:, 0], [
        X[:, 0] <= intersection_point_opt,
        X[:, 0] > intersection_point_opt], [
                                 lambda x: constant_func_opt,
                                 lambda x:
                                 linear_decline_slope_opt * x +
                                 constant_func_opt]),
             label='Fitted Model', color='red')
    plt.scatter(X, y, label='Data')
    plt.title(name)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #help
    # main()
    path_to_samples: str = sys.argv[1]
    path_to_responses: str = sys.argv[2]
    name_of_output_file: str = sys.argv[3]
    f: pd.DataFrame = pd.read_csv(path_to_samples, delimiter='\t', index_col=0,
                                        names=['id', 'seq'])
    responses: pd.DataFrame = pd.read_csv(path_to_responses, delimiter='\t',
                                          index_col=0,
                                          names=['id', 'half life 0A',
                                                 'half life 40A'])
    deg_rate_from_half_life: pd.Series = np.log(2) / responses['half life 0A']
    log2_of_deg_rate: pd.Series = np.log2(deg_rate_from_half_life)
    responses['degradation rate'] = round(log2_of_deg_rate, 4)
    responses['x0'] = 0
    responses['t0'] = 0
    kmer_matrix = matrix_generator(f, responses, 3, 7)
    kmer_matrix.to_csv(f"{name_of_output_file}.csv", index=True)
