from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_sum = 0
    validation_sum = 0
    shuffle_range = range(X.shape[0])
    splited_bins = np.array_split(shuffle_range, cv)
    for bin_ in splited_bins:
        train_negs = np.setdiff1d(shuffle_range, bin_)
        est_copy = deepcopy(estimator)
        fitted_estimator = est_copy.fit(X[train_negs], y[train_negs])
        train_sum += scoring(y[train_negs], fitted_estimator.predict(X[train_negs]))
        validation_sum += scoring(y[bin_], fitted_estimator.predict(X[bin_]))
    train_score_avg,validation_score_avg = train_sum / cv,  validation_sum / cv
    return train_score_avg, validation_score_avg














