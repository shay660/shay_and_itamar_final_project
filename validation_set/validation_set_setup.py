from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


def find_best_parameter(X: np.array, y: pd.Series) -> Tuple[int, List[Tuple[
    LinearRegression, LinearRegression or None]], np.array]:
    """
    find the best parameter for the step function (the t0 parameter).
    :param X: np.array from 1 to 10
    :param y: mRNA levels in each hour.
    :return: the best to and all the models that were check.
    """
    errors: List[float] = []
    models_lst: List[Tuple[LinearRegression, LinearRegression or None]] = []
    predictions: List[np.array] = []
    for t0 in range(9):
        if t0 == 0:
            y_pred, model1, model2 = linear_without_step(y, X)
        else:
            y_pred, model1, model2 = linear_with_step(t0, y, X)
            # linear with step a slope = 0  for the first model
            y_pred_const_func, model1_const_func, model2_const_func = \
                linear_with_step_constant(X, y, t0)

            if y_pred_const_func is not None:
                first_error = mean_squared_error(y, y_pred_const_func)
                if y_pred is not None:
                    sec_error = mean_squared_error(y, y_pred)
                    if first_error <= sec_error:
                        y_pred = y_pred_const_func
                else:
                    y_pred = y_pred_const_func
        if y_pred is not None:
            errors.append(mean_squared_error(y, y_pred))
            predictions.append(y_pred)

        else:
            errors.append(np.inf)
            predictions.append(y_pred)

        models_lst.append((model1, model2))
    t0 = int(np.argmin(errors))
    return t0, models_lst, predictions[t0]


def linear_with_step(t0: int, y: pd.Series, X: np.array) -> Tuple[
    np.ndarray or None, LinearRegression, LinearRegression]:
    """
    fit a model and calculate the model prediction for a specific t0 value.
    :param t0:
    :param y:
    :param X:
    :return: The model prediction two fitted models for this t0.
    """
    model1 = LinearRegression()
    model1.fit(X[:t0], y[:t0])
    model2 = LinearRegression()
    model2.fit(X[t0:], y[t0:])
    prediction1 = model1.predict(X[:t0])
    prediction2 = model2.predict(X[t0:])
    if prediction1[-1] <= prediction2[0]:
        return None, model1, model2
    y_pred = np.concatenate((prediction1, prediction2))
    return y_pred, model1, model2


def linear_without_step(y: pd.Series, X: np.array) -> Tuple[
    np.ndarray, LinearRegression, None]:
    """

    :param y:
    :param X:
    :return:
    """
    model1 = LinearRegression()
    model1.fit(X, y)
    y_pred = model1.predict(X)
    return y_pred, model1, None

def linear_with_step_constant(X: np.array, y: pd.Series, t0: int) -> Tuple[
    np.array or None, None, LinearRegression]:
    model2 = LinearRegression()
    model2.fit(X[t0:], y[t0:])
    prediction1: List[float] = [np.average(y[:t0]) for _ in range(t0)]
    prediction2 = model2.predict(X[t0:])
    y_pred = np.concatenate((prediction1, prediction2))
    assert len(y_pred) == 9
    if prediction1[-1] <= prediction2[0]:
        return None, None, model2

    return y_pred, None, model2


def predict(step_loc: int,
            models: List[Tuple[LinearRegression, LinearRegression]],
            X_pred) -> np.ndarray:
    if step_loc == 0:
        # If there is no step, use the single model and predict the entire X_pred.
        best_model = models[step_loc][0]
        y_pred = best_model.predict(X_pred)
    else:
        # If there is a step, split the prediction into two parts.
        best_model1, best_model2 = models[step_loc]
        first_part_pred = best_model1.predict(X_pred[:step_loc])
        sec_part_pred = best_model2.predict(X_pred[step_loc:])

        # Check if the last prediction of the first model is less than or equal to the first prediction of the second model.
        if first_part_pred[-1] <= sec_part_pred[0]:
            step_loc = 0  # Set step_loc back to 0
            best_model = models[step_loc][0]
            y_pred = best_model.predict(X_pred)
        else:
            y_pred = np.concatenate((first_part_pred, sec_part_pred))

    return y_pred

def make_plot(X, true_y, y_pred_plot, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, true_y, label='Data', color='blue')
    plt.plot(X, y_pred_plot, color='grey')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()