import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from sklearn.metrics import roc_curve, auc
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.metrics import misclassification_error
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
import plotly.express as px


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values_lst = []
    weights_lst = []
    def call(solver, weights, val, grad, t, eta, delta):
        weights_lst.append(weights)
        values_lst.append(val)

    return call, values_lst, weights_lst


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        for i, l in enumerate([L1, L2]):
            callback, vals_lst, weights_lst = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            gd.fit(l(init), None, None)
            plot_descent_path(l, np.asarray(weights_lst), f"norm = L{i + 1} "
                                                      f"and eta = "
                                                      f"{eta}").show()
            px.line(x=list(range(len(vals_lst))), y=vals_lst,
                    title=f"convergence   L{i + 1} , eta = "
                          f"{eta}").show()
            print(f" lowest lost of L{i + 1} , eta {eta} is :"
                  f"{np.round(min(vals_lst), 3)}")

    print("*" * 30 + '\n')

def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    # raise NotImplementedError()
    #
    # # Plot algorithm's convergence for the different values of gamma
    # raise NotImplementedError()
    #
    # # Plot descent path for gamma=0.95
    # raise NotImplementedError()
    pass

def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    gd = GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4))
    logistic_reg = LogisticRegression(solver=gd).fit(np.asarray(X_train),
                                                           np.asarray(y_train))
    y_prob = logistic_reg.predict_proba(X=np.asarray(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                          line=dict(color="black", dash='dash'),
                          name="Random Class Assignment"),
               go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                          name="1", showlegend=False, marker_size=5,
                          hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
              layout=go.Layout(
                  title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                  xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                  yaxis=dict(
                      title=r"$\text{True Positive Rate (TPR)}$"))).show()
    best_alpha = np.round(thresholds[np.argmax(tpr - fpr)], 3)
    print(f"The best alpha is {best_alpha}\n")
    print("*" * 30)

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    error_lst = []

    for i in [1, 2]:
        for lam in lambdas:
            logistic_solver = LogisticRegression(penalty=f"l{i}", alpha=0.5,
                                                 lam=lam, solver=gd)
            err_validation = cross_validate(logistic_solver,
                                            np.asarray(X_train),
                                            np.asarray(y_train),
                                            scoring=misclassification_error)[1]
            error_lst.append(err_validation)
        best_lam = lambdas[np.argmin(error_lst)]
        best_lam_model = LogisticRegression(penalty=f"l{i}", alpha=0.5,
                                            solver=gd,
                                            lam=best_lam).fit(
            np.asarray(X_test), np.asarray(y_test))
        best_model_loss = round(
            best_lam_model.loss(np.asarray(X_test), np.asarray(y_test)), 3)
        print(f"Optimal lambda when penalty is L{i} = {best_lam}")
        print(f"Worst error over the test set of the model = {best_model_loss}\n")
        print("*" * 30)


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
