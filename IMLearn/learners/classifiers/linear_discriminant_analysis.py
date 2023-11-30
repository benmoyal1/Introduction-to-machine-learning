from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        labels, count = np.unique(y, return_counts=True)
        self.classes_ = labels
        self.pi_ = count / len(y)

        means = list()
        for clas in self.classes_:
            means.append(np.mean(X[y == clas], axis=0))
        self.mu_ = np.array(means)
        n_samples, d_features = X.shape
        k = len(self.classes_)
        cov_init = np.zeros((d_features, d_features))
        c = X - self.mu_[y.astype(int)]
        for i in range(n_samples):
            for j in range(d_features):
                for k in range(d_features):
                    cov_init[j, k] += c[i, j] * c[i, k]
        cov_init /= n_samples - k
        self.cov_ = cov_init
        self.cov_inv_ = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        max_ind = np.argmax(self.likelihood(X), axis=1)
        return self.classes_[max_ind]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        difference_from_mean = X[:, np.newaxis, :] - self.mu_
        normalization_factor = np.sqrt((2 * np.pi) ** X.shape[1] * np.linalg.det(self.cov_))
        exponent = np.exp(-0.5 * np.sum(difference_from_mean.dot(self.cov_inv_) * difference_from_mean, axis=2))
        likelihoods = exponent / normalization_factor * self.pi_

        return likelihoods

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
        pred = self.predict(X)
        return misclassification_error(y, pred)
