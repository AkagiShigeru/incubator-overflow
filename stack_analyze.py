#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Tools to analyze posts (e.g. user input) based on fitted models and processing routines
#
#
import os
import dill
from collections import defaultdict

import numpy as np
import pandas as pd

from stack_util import local_import, UnescapeHTML, TextCleansing
from stack_words import GetRelevantWords
from stack_readin import own_cols

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


def GetAllFeatures(userposts, cfg, debug=False):
    """ Calculate features used by models / estimators.
        Beware: features could be hard-coded here with respect to other streamlined anlaysis scripts."""

    wdict = cfg.data["wdict"]

    features = defaultdict(list)
    for userpost in userposts:

        features["Body_unesc"].append(UnescapeHTML(userpost["Body"].decode("utf-8")))

        # raw features in stack_readin.py
        for own_col, own_fct in own_cols.items():

            features[own_col].append(own_fct({"Body_unesc": features["Body_unesc"][-1]}))

        # from stack_nlp.py in PrepareData
        features["titlelen"].append(len(userpost["Title"]))

        # from stack_nlp.py in PrepareData
        d = pd.to_datetime(userpost["CreationDate"])
        features["dayhour"].append(d.hour)
        features["weekday"].append(d.dayofweek)
        features["day"].append(d.dayofyear)

        # from stack_words.py
        ws, nws, ratio = GetRelevantWords(features["Body_unesc"][-1], get_ratio=True)

        if debug:
            print ws, nws, ratio

        wsdf = pd.DataFrame({"w": ws.keys(), "mult": ws.values()})
        wsdf.set_index("w", inplace=True, drop=False)
        wsdf = wsdf.join(wdict, how="inner")

        features["ratio"].append(float(ratio))
        features["nwords"].append(int(nws))
        features["ordersum"].append(float(wsdf.order.sum()) if not np.isnan(wsdf.order.sum()) else 0.)
        features["ordermean"].append(float(wsdf.order.mean()) if not np.isnan(wsdf.order.mean()) else 0.)
        features["orderstd"].append(float(wsdf.order.std()) if not np.isnan(wsdf.order.std()) else 0.)

    return features


def AnalyzePost(cfg, userposts=None, pids=None):
    """
    Analyze an existing post in db or custom user input.
    """
    if userposts is not None:
        print "Analyzing posts provided by user"
        print userpost
    elif pids is not None:
        print "Taking posts from db and caches..."
        # userpost = ...
        # todo implement
    else:
        AssertionError("No input provided!")

    # feature calculation
    post_features = GetAllFeatures(userposts, cfg)
    print post_features

    for fitcfg in cfg.fits:

        fneeds = fitcfg["feature_needs"]
        tokenizer = fitcfg["tokenizer_obj"]
        model = fitcfg["model_obj"]

        allfeatures = []
        if "posts" in fneeds:

            posts = post_features["Body_unesc"]
            if fitcfg.get("clean", False):
                posts = [TextCleansing(p) for p in posts]

            posts = tokenizer.texts_to_sequences(posts)
            maxlen_posts = 500  # this catches roughly 95 % of all posts with text cleaning
            posts = pad_sequences(posts, maxlen=maxlen_posts, padding="post", truncating="post")

            allfeatures.append(posts)

        if "titles" in fneeds:

            titles = post_features["Body_unesc"]
            if fitcfg.get("clean", False):
                titles = [TextCleansing(t) for t in titles]

            titles = tokenizer.texts_to_sequences(titles)
            maxlen_titles = 30  # this catches roughly 95 % of all titles with text cleaning
            titles = pad_sequences(titles, maxlen=maxlen_titles, padding="post", truncating="post")

            allfeatures.append(titles)

        for fneed in fneeds:

            if fneed not in ["posts", "titles"]:

                allfeatures.append(post_features[fneed])

        print allfeatures




def PrepareModels(cfg):

    # loading word dictionary for feature calculation
    if "dict" not in cfg.data:
        store_dict = pd.HDFStore(cfg.dict_path, "r", complib="blosc", complevel=9)
        print "Loading word dictionary..."
        wdict = store_dict.select("dict")
        wdict["freqs"] = wdict.n * 1. / wdict.n.sum()
        wdict = wdict.sort_values(by="n", ascending=False)
        wdict["order"] = np.arange(1, wdict.shape[0] + 1)
        cfg.data["wdict"] = wdict

    for fitcfg in cfg.fits:

        print "Preparing %s." % fitcfg["id"]
        fitcfg["tokenizer_obj"] = dill.load(open(fitcfg.get("tokenizer", "./models/tokenizer_%s.dill" % fitcfg["id"]), "r"))
        fitcfg["model_obj"] = load_model("./models/keras_full_%s.keras" % fitcfg["id"])

        # which features are needed
        feature_needs = []
        if fitcfg.get("posts", False):
            feature_needs.append("posts")
        if fitcfg.get("titles", False):
            feature_needs.append("titles")
        feature_needs.extend(fitcfg.get("cat_features", []))
        feature_needs.extend(fitcfg.get("features", []))
        fitcfg["feature_needs"] = feature_needs

    return cfg


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("Stackoverflow post analysis steered by config file.")
    parser.add_argument("fn", nargs=1, metavar="<config file>",
                        help="Configuration file containing desired paths and settings.")
    args = parser.parse_args()

    cfgfn = args.fn[0]

    # importing things from file
    cfg = local_import(cfgfn)

    # making sure that relevant directories exist
    for k, p in cfg.paths.items():
        if not os.path.exists(p):
            print "Creating non-existing directory {0}.".format(p)
            os.makedirs(p)

    userpost = {"Body": "This is crappy testpost",
                "Title": "testpost",
                "CreationDate": "26/11/2017",
                "UserName": "testuser"}

    cfg = PrepareModels(cfg)
    AnalyzePost(cfg, userposts=[userpost], pids=None)
