from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.fitted_ = True
        error, separation, signs, threshold, num_cols = np.inf, -1, 0, 0, len(X[0])
        possibilities = [-1, 1]
        product_gathered = product(range(num_cols), possibilities)
        for j, label in product_gathered:
            cur_threshold, cur_error = self._find_threshold(X[:, j], y, label)
            if cur_error < error:
                error, separation, signs, threshold = cur_error, j, label, cur_threshold
        self.j_ = separation
        self.threshold_ = threshold
        self.sign_ = signs

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        prediction = []
        to_classifie = X.T[self.j_]
        for samp in to_classifie:
            if samp >= self.threshold_:
                prediction.append(self.sign_)
            else:
                prediction.append(-self.sign_)

        return np.array(prediction)


def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
    """
    Given a feature vector and labels, find a threshold by which to perform a split
    The threshold is found according to the value minimizing the misclassification
    error along this feature

    Parameters
    ----------
    values: ndarray of shape (n_samples,)
        A feature vector to find a splitting threshold for

    labels: ndarray of shape (n_samples,)
        The labels to compare against

    sign: int
        Predicted label assigned to values equal to or above threshold

    Returns
    -------
    thr: float
        Threshold by which to perform split

    thr_err: float between 0 and 1
        Misclassificaiton error of returned threshold

    Notes
    -----
    For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
    which equal to or above the threshold are predicted as `sign`
    """

    sorted_indices = np.argsort(values)
    sorted_values, sorted_labels = values[sorted_indices], labels[sorted_indices]
    absolute_labels = np.abs(sorted_labels)
    loss = np.sum(absolute_labels[np.sign(sorted_labels) == sign])
    loss = np.append(loss, loss - np.cumsum(sorted_labels * sign))
    min_loss_index = np.argmin(loss)
    thresholds = np.concatenate([[-np.inf], sorted_values[1:], [np.inf]])
    threshold = thresholds[min_loss_index]
    loss_value = loss[min_loss_index]
    return threshold, loss_value


def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
    """
    Evaluate performance under misclassification loss function

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Test samples

    y : ndarray of shape (n_samples, )
        True labels of test samples

    Returns
    -------
    loss : float
        Performance under missclassification loss function
    """
    from ...metrics import misclassification_error
    return misclassification_error(y, self._predict(X))
