#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Default config file for stack words analysis.
#  For a custom analysis, make a copy of this default config file and work from there.
#
#
import os
from copy import copy
import numpy as np


# paths
paths = {}
paths["caches"] = "/home/alex/data/stackexchange/overflow/caches/"
paths["db"] = os.path.join(paths["caches"], "posts.db")
paths["metas"] = paths["caches"]
paths["features"] = os.path.join(paths["caches"], "features_new/")
paths["dictionaries"] = os.path.join(paths["caches"], "dictionaries/")


# data
data = {}

posts_path = paths["db"]
meta_path = os.path.join(paths["metas"], "posts_2017.hdf5")
# dict_path = os.path.join(paths["dictionaries"], "words_2017.hdf5")  # for old features
dict_path = os.path.join(paths["dictionaries"], "merged.hdf5")
features_path = os.path.join(paths["features"], "features_2017.hdf5")


# options (what data to read etc)
options = {}
options["read"] = ["questions", "features"]

# add other fit types later
# defining fits
fits = []
fit_nn = {}
fit_nn["id"] = "keras_tagprediction"
fit_nn["type"] = "keras_embedding"
fit_nn["name"] = "Predicting code topics / tags with word embeddings"
fit_nn["embed_dim"] = 300
fit_nn["embed_path"] = "/home/alex/data/glove.6B.%id.txt" % fit_nn["embed_dim"]
fit_nn["embed_out"] = "./glove.6B.%id.txt.word2vec" % fit_nn["embed_dim"]
fit_nn["nfeatures"] = 20000
fit_nn["posts"] = True
fit_nn["titles"] = True
# fit_nn["features"] = ["BodyNCodes", "BodyNQMarks",
#                       "BodySize", "titlelen", "nwords", "ordermean",
#                       "orderstd", "ratio", "weekday", "dayhour", "day"]
fit_nn["features"] = ["BodyNCodes", "BodyNQMarks",
                      "BodySize", "titlelen", "nwords", "ordermean",
                      "orderstd", "weekday", "dayhour", "day"]

fit_nn["labelfct"] = lambda df: np.asarray(df.Tags.apply(lambda x: "python" in x))
# fit_nn["labelfct"] = np.asarray(qs.Score > 0, dtype=int)

fit_nn["nsample"] = 200000
fit_nn["uniform"] = True
fit_nn["nepoch"] = 5
fit_nn["nbatch"] = 100
fit_nn["nsplit"] = 0.2
fit_nn["save"] = True
fits.append(fit_nn.copy())
