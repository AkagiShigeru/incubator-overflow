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
paths["caches"] = "/home/ubuntu/data/stack/"
paths["db"] = os.path.join(paths["caches"], "posts.db")
paths["metas"] = paths["caches"]
paths["features"] = os.path.join(paths["caches"], "features_new/")
paths["dictionaries"] = os.path.join(paths["caches"], "dictionaries/")


# data
data = {}

# TO DO: functionality to run multiple years at once
year = 2016

posts_path = paths["db"]
meta_path = os.path.join(paths["metas"], "posts_%s.hdf5" % year)
dict_path = os.path.join(paths["dictionaries"], "merged.hdf5")
features_path = os.path.join(paths["features"], "features_%s.hdf5" % year)
mostcommontags_path = "./infos/most_common_tags.csv"
mostcommon_tags = pd.read_csv(mostcommontags_path)
feat_quantiles_path = "./infos/feature_quantiles.csv"
feat_quantiles = pd.read_csv(feat_quantiles_path)

# options (what data to read etc)
options = {}
options["read"] = ["questions", "features"]

# add other fit types later
# defining fits
fits = []


fit_nn = {}
fit_nn["id"] = "keras_tagprediction"
fit_nn["type"] = "keras_embedding_tags"
fit_nn["name"] = "Predicting code topics / tags with word embeddings"
fit_nn["embed_dim"] = 300
fit_nn["embed_path"] = "/home/ubuntu/data/stack/glove.6B.%id.txt" % fit_nn["embed_dim"]
fit_nn["embed_out"] = "./glove.6B.%id.txt.word2vec" % fit_nn["embed_dim"]
fit_nn["nfeatures"] = 50000
fit_nn["posts"] = True
fit_nn["titles"] = True
fit_nn["features"] = ["BodyNCodes", "BodyNQMarks",
                      "BodySize", "titlelen", "nwords", "ordermean",
                      "orderstd"]
fit_nn["cat_features"] = ["weekday", "dayhour"]

# just identifying python label
# fit_nn["labelfct"] = lambda df: np.asarray(df.Tags.apply(lambda x: "python" in x))


def LocateFirst(l, tagdf, nt=10):
    """ Finds occurence of hottest tag: first priority is order in l and then order in tagdf. """
    ins = np.isin(l, tagdf.iloc[:nt].tags.values)
    first = np.where(ins)[0]
    if len(first) > 0:
        return np.where(tagdf.iloc[:nt].tags.values == l[first[0]])[0][0]
    return nt


# for later
def LocateAll(l, tagdf, nt=10):
    ins = np.isin(tagdf.iloc[:nt].tags.values, l)
    return np.asarray(list(ins) + [1 if np.sum(ins) > 0 else 0], dtype=int)

# identifying first n labels
fit_nn["ntags"] = 30
fit_nn["labelfct"] = lambda df, fcfg: np.asarray(df.Tags.apply(lambda x: LocateFirst(np.asarray(x), mostcommon_tags, fcfg["ntags"])))

fit_nn["grouplabels"] = list(mostcommon_tags.iloc[:fit_nn["ntags"]].tags.values) + ["other"]
fit_nn["nsample"] = 500000
fit_nn["seed"] = 42
fit_nn["uniform"] = False
fit_nn["nepoch"] = 10
fit_nn["nbatch"] = 100
fit_nn["nsplit"] = 0.2
fit_nn["save"] = True
fit_nn["binary"] = False
fit_nn["clean"] = True
fit_nn["cnn"] = False
fit_nn["train_embeddings"] = True
fit_nn["from_cache"] = True
# fits.append(fit_nn.copy())


fit_nn = {}
fit_nn["id"] = "keras_tagprediction_cnn"
fit_nn["type"] = "keras_embedding_tags"
fit_nn["name"] = "Predicting code topics / tags with word embeddings"
fit_nn["embed_dim"] = 300
fit_nn["embed_path"] = "/home/ubuntu/data/stack/glove.6B.%id.txt" % fit_nn["embed_dim"]
fit_nn["embed_out"] = "./glove.6B.%id.txt.word2vec" % fit_nn["embed_dim"]
fit_nn["nfeatures"] = 50000
fit_nn["posts"] = True
fit_nn["titles"] = True
fit_nn["use_saved_posts"] = True
fit_nn["use_saved_events"] = False
fit_nn["features"] = ["BodyNCodes", "BodyNQMarks",
                      "BodySize", "titlelen", "nwords", "ordermean",
                      "orderstd"]
fit_nn["cat_features"] = ["weekday", "dayhour"]

# identifying first n labels
fit_nn["ntags"] = 30
fit_nn["labelfct"] = lambda df, fcfg: np.asarray(df.Tags.apply(lambda x: LocateFirst(np.asarray(x), mostcommon_tags, fcfg["ntags"])))

fit_nn["grouplabels"] = list(mostcommon_tags.iloc[:fit_nn["ntags"]].tags.values) + ["other"]
fit_nn["nsample"] = 400000
fit_nn["seed"] = 42
fit_nn["uniform"] = False
fit_nn["nepoch"] = 100
fit_nn["nbatch"] = 100
fit_nn["nsplit"] = 0.2
fit_nn["save"] = True
fit_nn["binary"] = False
fit_nn["clean"] = True
fit_nn["dropout"] = False
fit_nn["cnn"] = True
fit_nn["premade_embeddings"] = True
fit_nn["train_embeddings"] = True
fit_nn["from_cache"] = False
fits.append(fit_nn.copy())


fit_nn = {}
fit_nn["id"] = "keras_tagprediction_cnn_notitles"
fit_nn["type"] = "keras_embedding_tags"
fit_nn["name"] = "Predicting code topics / tags with word embeddings"
fit_nn["embed_dim"] = 300
fit_nn["embed_path"] = "/home/ubuntu/data/stack/glove.6B.%id.txt" % fit_nn["embed_dim"]
fit_nn["embed_out"] = "./glove.6B.%id.txt.word2vec" % fit_nn["embed_dim"]
fit_nn["nfeatures"] = 50000
fit_nn["posts"] = True
fit_nn["titles"] = False
fit_nn["use_saved_posts"] = True
fit_nn["use_saved_events"] = False
fit_nn["features"] = ["BodyNCodes", "BodyNQMarks",
                      "BodySize", "titlelen", "nwords", "ordermean",
                      "orderstd"]
fit_nn["cat_features"] = ["weekday", "dayhour"]

# identifying first n labels
fit_nn["ntags"] = 30
fit_nn["labelfct"] = lambda df, fcfg: np.asarray(df.Tags.apply(lambda x: LocateFirst(np.asarray(x), mostcommon_tags, fcfg["ntags"])))

fit_nn["grouplabels"] = list(mostcommon_tags.iloc[:fit_nn["ntags"]].tags.values) + ["other"]
fit_nn["nsample"] = 400000
fit_nn["seed"] = 42
fit_nn["uniform"] = False
fit_nn["nepoch"] = 100
fit_nn["nbatch"] = 100
fit_nn["nsplit"] = 0.2
fit_nn["save"] = True
fit_nn["binary"] = False
fit_nn["clean"] = True
fit_nn["dropout"] = False
fit_nn["cnn"] = True
fit_nn["premade_embeddings"] = False
fit_nn["train_embeddings"] = True
fit_nn["from_cache"] = False
# fits.append(fit_nn.copy())


fit_nn = {}
fit_nn["id"] = "keras_scoreprediction_twoclasses"
fit_nn["type"] = "keras_embedding_scores"
fit_nn["name"] = "Predicting questions scores with word embeddings"
fit_nn["embed_dim"] = 300
fit_nn["embed_path"] = "/home/ubuntu/data/stack/glove.6B.%id.txt" % fit_nn["embed_dim"]
fit_nn["embed_out"] = "./glove.6B.%id.txt.word2vec" % fit_nn["embed_dim"]
fit_nn["nfeatures"] = 50000
fit_nn["posts"] = True
fit_nn["titles"] = True
fit_nn["features"] = ["BodyNCodes", "BodyNQMarks",
                      "BodySize", "titlelen", "nwords", "ordermean",
                      "orderstd"]
fit_nn["cat_features"] = ["weekday", "dayhour"]


def scoregroups(df, upqs=[0.1, 0.9]):
    from scipy.stats.mstats import mquantiles
    df["label"] = 1
    print "Performing automagical score grouping"
    upqvals = np.append([df.Score.min()], np.append(mquantiles(df.Score, prob=upqs), [df.Score.max()]))
    for ui in xrange(len(upqvals) - 1):
        print "Group %i: score range: [%.1f, %.1f)" % (ui, upqvals[ui], upqvals[ui + 1])
        df.loc[(df.Score >= upqvals[ui]) & (df.Score < upqvals[ui + 1]), "label"] = ui
    return df.label

# score groups
fit_nn["labelfct"] = lambda df, fcfg: scoregroups(df, upqs=[0.96])
fit_nn["grouplabels"] = ["normal", "good"]
fit_nn["tokenizer"] = "./models/tokenizer_keras_tagprediction.dill"
fit_nn["nsample"] = 200000
fit_nn["seed"] = 42
fit_nn["uniform"] = True
fit_nn["nepoch"] = 10
fit_nn["nbatch"] = 100
fit_nn["nsplit"] = 0.2
fit_nn["save"] = True
fit_nn["binary"] = False
fit_nn["clean"] = True
fit_nn["cnn"] = False
fit_nn["train_embeddings"] = True
fit_nn["from_cache"] = True
# fits.append(fit_nn.copy())


fit_nn = {}
fit_nn["id"] = "keras_scoreprediction_twoclasses_cnn"
fit_nn["type"] = "keras_embedding_scores"
fit_nn["name"] = "Predicting questions scores with word embeddings"
fit_nn["embed_dim"] = 300
fit_nn["embed_path"] = "/home/ubuntu/data/stack/glove.6B.%id.txt" % fit_nn["embed_dim"]
fit_nn["embed_out"] = "./glove.6B.%id.txt.word2vec" % fit_nn["embed_dim"]
fit_nn["nfeatures"] = 50000
fit_nn["posts"] = True
fit_nn["titles"] = True
fit_nn["use_saved_posts"] = True
fit_nn["features"] = ["BodyNCodes", "BodyNQMarks",
                      "BodySize", "titlelen", "nwords", "ordermean",
                      "orderstd"]
fit_nn["cat_features"] = ["weekday", "dayhour"]

# score groups
fit_nn["labelfct"] = lambda df, fcfg: scoregroups(df, upqs=[0.96])
fit_nn["grouplabels"] = ["normal", "good"]
fit_nn["tokenizer"] = "./models/tokenizer_keras_tagprediction.dill"
fit_nn["nsample"] = 200000
fit_nn["seed"] = 42
fit_nn["uniform"] = True
fit_nn["nepoch"] = 10
fit_nn["nbatch"] = 100
fit_nn["nsplit"] = 0.2
fit_nn["save"] = True
fit_nn["binary"] = False
fit_nn["clean"] = True
fit_nn["cnn"] = True
fit_nn["dropout"] = True
fit_nn["train_embeddings"] = True
fit_nn["from_cache"] = True
# fits.append(fit_nn.copy())


fit_nn = {}
fit_nn["id"] = "keras_scoreprediction"
fit_nn["type"] = "keras_embedding_scores"
fit_nn["name"] = "Predicting questions scores with word embeddings"
fit_nn["embed_dim"] = 300
fit_nn["embed_path"] = "/home/ubuntu/data/stack/glove.6B.%id.txt" % fit_nn["embed_dim"]
fit_nn["embed_out"] = "./glove.6B.%id.txt.word2vec" % fit_nn["embed_dim"]
fit_nn["nfeatures"] = 50000
fit_nn["posts"] = True
fit_nn["titles"] = True
fit_nn["features"] = ["BodyNCodes", "BodyNQMarks",
                      "BodySize", "titlelen", "nwords", "ordermean",
                      "orderstd"]
fit_nn["cat_features"] = ["weekday", "dayhour"]

# just identifying python label
# fit_nn["labelfct"] = lambda df: np.asarray(df.Tags.apply(lambda x: "python" in x))

# score groups
fit_nn["labelfct"] = lambda df, fcfg: scoregroups(df, upqs=[0.1, 0.9, 0.97])
fit_nn["grouplabels"] = ["bad", "normal", "good", "very good"]
fit_nn["tokenizer"] = "./models/tokenizer_keras_tagprediction.dill"
fit_nn["nsample"] = 200000
fit_nn["seed"] = 42
fit_nn["uniform"] = False
fit_nn["nepoch"] = 10
fit_nn["nbatch"] = 100
fit_nn["nsplit"] = 0.2
fit_nn["save"] = True
fit_nn["binary"] = False
fit_nn["clean"] = True
fit_nn["cnn"] = False
fit_nn["train_embeddings"] = True
fit_nn["from_cache"] = True
# fits.append(fit_nn.copy())
