#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Custom scikit learn methods / transformers etc
#
#
import numpy as np
# import pandas as pd

from sklearn import base

from collections import Counter


class CustomCombiner(base.BaseEstimator, base.TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.around(np.mean(X, axis=0))


class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, col_names):
        self.col_names = col_names  # We will need these in transform()

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self

    def transform(self, X):
        return np.squeeze(X[self.col_names].values)


class DictEncoder(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        enc = []
        for row in X:
            enc.append(Counter(row))
        return enc


class EstimatorTransformer(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        # Use predict on the stored estimator as a "transformation".
        # Be sure to return a 2-D array.
        return [[x] for x in self.estimator.predict(X)]


# custom regressor (*heavy sweat in face while watching multiple regressions*)
class MultiModelRegressor(base.BaseEstimator, base.RegressorMixin):

    def __init__(self, base_est, resi_est):
        self.base_est = base_est
        self.resi_est = resi_est
        self.base_pred = None
        self.best_pred = None

    def fit(self, X, y):
        self.base_est.fit(X, y)
        print "Performing base regression..."
        self.base_pred = self.base_est.predict(X)
        resis = y - self.base_pred
        print "Performing residual regression..."
        self.resi_est.fit(X, resis)
        # self.best_pred = self.resi_est(X)
        return self

    def predict(self, X):
        return self.base_est.predict(X) + self.resi_est.predict(X)
