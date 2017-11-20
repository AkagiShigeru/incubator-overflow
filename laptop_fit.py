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
import pandas as pd


# paths
paths = {}
paths["caches"] = "/home/alex/data/stackexchange/overflow/caches/"
paths["db"] = os.path.join(paths["caches"], "posts.db")
paths["metas"] = paths["caches"]
paths["features"] = os.path.join(paths["caches"], "features_new/")
paths["dictionaries"] = os.path.join(paths["caches"], "dictionaries/")


# data
data = {}

year = 2016

posts_path = paths["db"]
meta_path = os.path.join(paths["metas"], "posts_%s.hdf5" % year)
dict_path = os.path.join(paths["dictionaries"], "merged.hdf5")
features_path = os.path.join(paths["features"], "features_%s.hdf5" % year)
mostcommontags_path = "./infos/most_common_tags.csv"
mostcommon_tags = pd.read_csv(mostcommontags_path)

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
                      "orderstd"]
fit_nn["cat_features"] = ["weekday", "dayhour", "day"]

# just identifying python label
# fit_nn["labelfct"] = lambda df: np.asarray(df.Tags.apply(lambda x: "python" in x))


def LocateFirst(l, tagdf, nt=10):
    """ Returns index of most common element/tag in line of tags."""
    for e in xrange(nt):
        if tagdf.iloc[e].tags in l:
            return e
    else:
        return nt
# identifying first n labels
fit_nn["labelfct"] = lambda df: np.asarray(df.Tags.apply(lambda x: LocateFirst(x, mostcommon_tags, 20)))

# score > 0 label
# fit_nn["labelfct"] = np.asarray(qs.Score > 0, dtype=int)

fit_nn["nsample"] = 400000
fit_nn["uniform"] = False
fit_nn["nepoch"] = 20
fit_nn["nbatch"] = 100
fit_nn["nsplit"] = 0.2
fit_nn["save"] = True
fits.append(fit_nn.copy())
