import math
from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
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
    # train = X.sample(frac=train_proportion)
    # test = X.loc[X.index.difference(train.index)]
    # return train, y.loc[train.index], test, y.loc[test.index]

    samps_res = pd.concat([X, y], axis=1)
    mixed_all = samps_res.sample(frac=1)
    n = X.shape[0]
    train_len = math.ceil(train_proportion * n)
    train_df = mixed_all.iloc[:train_len]
    test_df = mixed_all.iloc[train_len:]
    train_x = train_df.iloc[:, :-1]
    train_y = train_df.iloc[:, -1]
    test_x = test_df.iloc[:, :-1]
    test_y = test_df.iloc[:, -1]
    return train_x,train_y,test_x,test_y
def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()

# sample = pd.DataFrame({
#     'feature_1': [1, 2, 3, 4, 5, 6, 7, 8],
#     'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
#     'feature_3': [-1, -2, -3, -4, -5, -6, -7, -8]
# })
#
# # Create a Series with 8 responses corresponding to the samples in the DataFrame
# y = pd.Series([0, 1, 0, 1, 0, 1, 1, 0])
#
# print(split_train_test(sample,y)[0])
# print(split_train_test(sample,y)[1])
# print(split_train_test(sample,y)[2])
# print(split_train_test(sample,y)[3])