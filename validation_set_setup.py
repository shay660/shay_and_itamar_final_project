import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def find_best_parameter(_X, y):
    _x0 = 0
    errors = []
    _models_lst = []
    for i in range(9):
        if _x0 == 0:
            _y_pred = linear_without_step(_models_lst, y, _X)
        else:
            _y_pred = linear_with_step(_models_lst, _x0,y, _X)
        _x0 += 1
        errors.append(mean_squared_error(y, _y_pred))
    _x0 = np.argmin(errors)
    return _x0, _models_lst


def linear_with_step(_models_lst, _x0, y, X_fit):
    model1 = LinearRegression()
    model1.fit(X_fit[:_x0], y[:_x0])
    model2 = LinearRegression()
    model2.fit(X_fit[_x0:], y[_x0:])
    prediction1 = model1.predict(X_fit[:_x0])
    prediction2 = model2.predict(X_fit[_x0:])
    y_pred = np.concatenate((prediction1, prediction2))
    _models_lst.append((model1, model2))
    return y_pred


def linear_without_step(_models_lst, y, _X):
    model1 = LinearRegression()
    model1.fit(_X, y)
    prediction1 = model1.predict(_X)
    y_pred = prediction1
    _models_lst.append(model1)
    return y_pred


def predict(step_loc, _models, X_pred):

    if step_loc != 0:
        best_model1, best_model2 = _models[step_loc][0], _models[step_loc][1]
        first_part_pred = best_model1.predict(X_pred[:step_loc])
        sec_part_pred = best_model2.predict(X_pred[step_loc:])
        if first_part_pred[-1] <= sec_part_pred[0]:
            step_loc = 0
            best_model = _models[step_loc]
            _y_pred = best_model.predict(X_pred)
            return _y_pred
        _y_pred = np.concatenate((first_part_pred, sec_part_pred))
        return _y_pred
    best_model = _models[step_loc]
    _y_pred = best_model.predict(X_pred)
    return _y_pred


def make_plot(x_plot, y_plot, y_pred_plot, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_plot, y_plot, label='Data', color='blue')
    plt.plot(x_plot, y_pred_plot, color='grey')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("data/normalized_mRNA_levels_Aplus.csv")
    x = np.arange(1, 10)
    X = np.array(range(1, 10)).reshape(-1, 1)

    for i, row in df.iterrows():
        x0, models_lst = find_best_parameter(X, row[1:])
        y_pred = predict(X_pred=X, step_loc=x0, _models=models_lst)
        # Plot the data and the selected model's predictions
        make_plot(x, row[1:], y_pred, row[0])

        print("Step Location step_loc:", x0)

    # intercepts = np.array([], dtype=float)
    # deg_rate = np.array([], dtype=float)
    # regression =LinearRegression()
    # X = np.array(range(1, 10)).reshape(-1, 1)
    # for i, row in df.iterrows():
    #     regression.fit(X, row[1:])
    #     intercepts = np.append(intercepts, regression.intercept_)
    #     deg_rate = np.append(deg_rate, regression.coef_[0])
    #
    # df["intercepts"] = intercepts
    # df["degradation rate"] = deg_rate
    # # pl.scatter(range(1,10), df.loc[0][1:10], label="Data Points")
    # # pl.plot(range(1,10), df.loc[0, "degradation rate"]*range(1,
    # #                                                          10) +
    # #         intercepts[0],
    # #          label="Dashed")
    # # pl.grid(True)
    # #
    # # plt.show()
    # # print(df)
    # sequences = pd.read_csv("data/Validation_sequences.csv")
    # df.to_csv("data/normalized_mRNA_levels_Aplus_with_deg_rate.csv")
    # matrix_generator(sequences, df, 3, 7).to_csv(
    #     "data/Validation_set_Aplus_kmer_count.csv")
