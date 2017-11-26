#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Tools to analyze posts (e.g. user input) based on fitted models and processing routines
#
#
import os
from stack_nlp import *
from stack_util import local_import


def GetAllFeatures(userposts, cfg):
    """ Calculate features used by models / estimators.
        Beware: features could be hard-coded here with respect to other streamlined anlaysis scripts."""

    from stack_readin import own_cols

    if "dict" not in cfg.data:
        store_dict = pd.HDFStore(cfg.dict_path, "r", complib="blosc", complevel=9)
        print "Loading word dictionary..."
        words = store_dict.select("dict")
        words["freqs"] = words.n * 1. / words.n.sum()
        words = words.sort_values(by="n", ascending=False)
        words["order"] = np.arange(1, words.shape[0] + 1)
    else:
        words = cfg["dict"]

    features = defaultdict(list)
    for userpost in userposts:

        # raw features in stack_readin.py
        for own_col, own_fct in own_cols.items():

            features[own_col].append(own_fct({"Body_unesc": UnescapeHTML(userpost["Body"].decode("utf-8"))}))

        # from stack_nlp.py in PrepareData
        features["titlelen"].append(len(userpost["Title"]))

        # from stack_nlp.py in PrepareData
        d = pd.to_datetime(userpost["CreationDate"])
        features["dayhour"].append(d.hour)
        features["weekday"].append(d.dayofweek)
        features["day"].append(d.dayofyear)

    return features


def AnalyzePost(cfg, userpost=None, pid=None):
    """
    Analyze an existing post in db or custom user input.
    """
    if userpost is not None:
        print "Analyzing post provided by user"
        print userpost
    elif pid is not None:
        print "Taking post from db and caches..."
        # userpost = ...
        # todo implement
    else:
        AssertionError("No input provided!")

    # feature calculation
    features = GetAllFeatures([userpost], cfg)
    print features


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

    AnalyzePost(cfg, userpost=userpost, pid=None)
