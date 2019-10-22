# Truncated Normal KDE
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2019, Yuriy Sverchkov

from functools import partial
from scipy.stats import truncnorm
from sklearn.base import BaseEstimator
import numpy as np


class TruncatedNormalKernelDensity(BaseEstimator):
    """Kernel Density Estimation on Bounded Support Using Truncated Normal Kernels

    Parameters
    ----------
    bandwidth : float
        The bandwidth of the kernel.

    lowerbound : float
        The lower-bound of the support.

    upperbound : float
        The upper-bound of the support.
    """
    def __init__(self,
                 bandwidth : float = 1.0,
                 lowerbound : float = -np.inf,
                 upperbound : float = np.inf):

        self.bandwidth = bandwidth
        self.lowerbound = lowerbound
        self.upperbound = upperbound

        if not bandwidth > 0:
            raise ValueError("Bandwidth must be positive.")

    def fit(self, X, y=None, sample_weight=None):
        """Fit the Kernel Density model on the data.

        Parameters
        ----------
        X : array_like, shape (n_samples,)
            list of data points.
        sample_weight : array_like, shape (n_samples,), optional
            list of sample weights attached to data X.
        """
        # X = check_array(X, order='C', dtype=DTYPE)
        self.points_ = X

        if sample_weight is None:
            weights = np.ones(shape=X.shape)
        else:
            # TODO Checks
            weights = sample_weight

        self.weights_ = weights
        self.normalizer_ = sum(weights)
        self.lowers_ = (self.lowerbound - self.points_) / self.bandwidth
        self.uppers_ = (self.upperbound - self.points_) / self.bandwidth

        return self

    def score_samples(self, X):
        return(
            sum([truncnorm.logpdf(X-mean, lower, upper) * weight for
                 (mean, lower, upper, weight) in
                 zip(self.points_, self.lowers_, self.uppers_, self.weights_)]
                ) / self.normalizer_)
