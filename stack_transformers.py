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
        return X[self.col_names]


class DictEncoder(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        a = X[self.col_name].apply(Counter)
        return a


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
