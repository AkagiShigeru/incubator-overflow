#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Tools to analyze posts (e.g. user input) based on fitted models and processing routines
#
#
import os
from stack_nlp import *
from stack_util import local_import
from stack_words import GetRelevantWords


def GetAllFeatures(userposts, cfg):
    """ Calculate features used by models / estimators.
        Beware: features could be hard-coded here with respect to other streamlined anlaysis scripts."""

    from stack_readin import own_cols

    if "dict" not in cfg.data:
        store_dict = pd.HDFStore(cfg.dict_path, "r", complib="blosc", complevel=9)
        print "Loading word dictionary..."
        wdict = store_dict.select("dict")
        wdict["freqs"] = wdict.n * 1. / wdict.n.sum()
        wdict = wdict.sort_values(by="n", ascending=False)
        wdict["order"] = np.arange(1, wdict.shape[0] + 1)
    else:
        wdict = cfg["dict"]

    features = []
    for userpost in userposts:

        feature_dict = {}

        feature_dict["Body_unesc"] = UnescapeHTML(userpost["Body"].decode("utf-8"))

        # raw features in stack_readin.py
        for own_col, own_fct in own_cols.items():

            feature_dict[own_col] = own_fct(feature_dict)

        # from stack_nlp.py in PrepareData
        feature_dict["titlelen"] = len(userpost["Title"])

        # from stack_nlp.py in PrepareData
        d = pd.to_datetime(userpost["CreationDate"])
        feature_dict["dayhour"] = d.hour
        feature_dict["weekday"] = d.dayofweek
        feature_dict["day"] = d.dayofyear

        # from stack_words.py
        ws, nws, ratio = GetRelevantWords(feature_dict["Body_unesc"], get_ratio=True)

        print ws, nws, ratio

        wsdf = pd.DataFrame({"w": ws.keys(), "mult": ws.values()})
        wsdf.set_index("w", inplace=True, drop=False)
        wsdf = wsdf.join(wdict, how="inner")

        feature_dict["ratio"] = float(ratio)
        feature_dict["nwords"] = int(nws)
        feature_dict["ordersum"] = float(wsdf.order.sum())
        feature_dict["ordermean"] = float(wsdf.order.mean())
        feature_dict["orderstd"] = float(wsdf.order.std())

        features.append(feature_dict)

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

    for fitcfg in cfg.fits:

        print fitcfg["id"]
        tokenizer = dill.load(open("./models/tokenizer_%s.dill" % fitcfg["id"], "r"))
        model = load_model("./models/keras_full_%s.keras" % fitcfg["id"])


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
