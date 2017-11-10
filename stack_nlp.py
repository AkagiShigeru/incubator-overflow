#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  NLP analysis of stackoverflow posts using conventional methods (no NN)
#
#
import os
import pandas as pd
import numpy as np

# import seaborn as sns
from pyik.mplext import ViolinPlot
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import sqlite3

from pyik.mplext import ViolinPlot

from stack_util import *
from stack_transformers import *

from sklearn.utils import shuffle

from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from scipy import stats
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder

from IPython import embed


def MergeDicts(dictlist):
    """ Return a merged dictionary. """
    pass


def SelectionAndShuffling(cfg):

    qs = cfg.data["meta"]
    qs = qs[qs.nwords > 5]

    # shuffling the df to avoid time ordering dependencies for now
    print "Shuffling data..."
    qs = shuffle(qs)
    print "Length of shuffled data:", len(qs)
    cfg.data["meta"] = qs


def PrepareData(cfg):

    store_meta = pd.HDFStore(cfg.meta_path, "r", complib="blosc", complevel=9)
    store_dict = pd.HDFStore(cfg.dict_path, "r", complib="blosc", complevel=9)
    store_feat = pd.HDFStore(cfg.features_path, "r", complib="blosc", complevel=9)

    # select only questions here
    smask = store_meta.select_as_coordinates("posts", "PostTypeId == 1")
    qs = store_meta.select("posts", where=smask)
    qs.set_index("Id", inplace=True, drop=False)
    print "Shape of question df", qs.shape

    # additional trivial columns / features, see also feature dataframe below
    qs["titlelen"] = qs["Title"].apply(len)
    # transforming tags
    qs["Tags"] = qs.Tags.apply(lambda x: x.split(";")[1:])
    qs["hasAnswers"] = qs.AnswerCount > 1

    # some datetime conv
    datecols = ["CreationDate"]
    for datecol in datecols:
        qs[datecol] = pd.to_datetime(qs[datecol], origin="julian", unit="D")

    now = pd.datetime.now()
    qs["dt_created"] = now - qs.CreationDate

    # getting the answers
    answers = store_meta.select("posts", where=store_meta.select_as_coordinates("posts", "PostTypeId == 2"))
    answers.set_index("Id", inplace=True, drop=False)
    print "Shape of answer df", answers.shape

    # word dictionary
    print "Loading word dictionary..."
    words = store_dict.select("dict")
    words["freqs"] = words.n * 1. / words.n.sum()
    words = words.sort_values(by="n", ascending=False)
    words["order"] = np.arange(1, words.shape[0] + 1)

    print "Shape of dictionary", words.shape

    # drop known nuisance words that made it into the list
    print "Warning! Dropping some words from word list, please verify!"
    drops = [1211]
    for dind in drops:
        print "Dropping %i" % dind
        words = words.drop(dind)

    features = store_feat.select("words")
    features.set_index("Id", inplace=True, drop=False)

    # join in information about occurring words, probabilities etc
    qs = qs.join(features, how="inner", rsuffix="_r")
    print "Shape of merged df", qs.shape

    mask = qs.nwords > 5
    print "Selecting only questions with at least 5 meaningful words."
    print "This removes %i questions." % (np.sum(~mask))
    qs = qs[mask]

    mask = (qs.ordersum.isnull()) | (qs.orderstd.isnull())
    print "Removing bad values with missing feature information."
    print "This affects %i questions." % (np.sum(mask))
    qs = qs[~mask]

    # merge information about first answer into the frame
    answers = answers.sort_values(by="CreationDate", ascending=True)
    qs = qs.merge(answers[["ParentId", "CreationDate"]].drop_duplicates("ParentId"),
                  how="left", left_on="Id", right_on="ParentId", suffixes=("", "_first"))

    # merge in information about accepted answer
    qs = qs.merge(answers[["Id", "CreationDate"]],
                  how="left", left_on="AcceptedAnswerId", right_on="Id", suffixes=("", "_acc"))

    qs["CreationDate_first"] = pd.to_datetime(qs.CreationDate_first, origin="julian", unit="D")
    qs["CreationDate_acc"] = pd.to_datetime(qs.CreationDate_acc, origin="julian", unit="D")

    qs["dayhour"] = qs.CreationDate.dt.hour
    qs["weekday"] = qs.CreationDate.dt.dayofweek

    # time between questions posing and first answer
    dtanswer = qs.CreationDate_first - qs.CreationDate
    dtanswer_acc = qs.CreationDate_acc - qs.CreationDate
    qs["dt_answer"] = dtanswer
    qs["dt_accanswer"] = dtanswer_acc

    qs["dt_answer_hour"] = qs.dt_answer.dt.total_seconds() * 1. / 3600
    qs["dt_accanswer_hour"] = qs.dt_accanswer.dt.total_seconds() * 1. / 3600

    # normalizing some columns
    print "Calculating normalized columns. They are available under usual column name + _norm."
    cols = ["BodyNCodes", "BodyNQMarks", "BodySize", "titlelen", "nwords", "ordersum",
            "ordermean", "orderstd", "ratio"]
    for col in cols:
        quants = qs[col].quantile([0.01, 0.99]).values
        qs["%s_norm" % col] = (qs[col] - quants[0]) / (quants[1] - quants[0])

    # saving for convenient use in notebooks
    cfg.data["meta"] = qs
    cfg.data["dict"] = words
    cfg.data["features"] = features
    cfg.data["answers"] = answers


def NormalizeColumns(df, cols):
    pass


def TimeAnalysis(cfg):
    """ Analyse time dependence of answers in more detail.
        Can be found in accompanying notebooks. """
    pass


def SimpleAnalysis(cfg):

    # selecting data from 2017
    # posts_path = os.path.join(cfg.paths["db"])
    meta_path = os.path.join(cfg.paths["metas"], "posts_2017.hdf5")
    dict_path = os.path.join(cfg.paths["dictionaries"], "words_2017.hdf5")
    features_path = os.path.join(cfg.paths["features"], "features_2017.hdf5")

    # conn = sqlite3.connect(posts_path)

    store_meta = pd.HDFStore(meta_path, "r", complib="blosc", complevel=9)
    store_dict = pd.HDFStore(dict_path, "r", complib="blosc", complevel=9)
    store_feat = pd.HDFStore(features_path, "r", complib="blosc", complevel=9)

    smask = store_meta.select_as_coordinates("posts", "PostTypeId == 1")
    qs = store_meta.select("posts", where=smask)
    qs.set_index("Id", inplace=True, drop=False)
    print "Shape of question df", qs.shape

    answers = store_meta.select("posts", where=store_meta.select_as_coordinates("posts", "PostTypeId == 2"))
    answers.set_index("Id", inplace=True, drop=False)
    print "Shape of answer df", answers.shape

    words = store_dict.select("all")

    words["freqs"] = words.n * 1. / words.n.sum()
    words = words.sort_values(by="n", ascending=False)
    words["order"] = np.arange(1, words.shape[0] + 1)
    # drop known nuisance words that made it into the list
    words = words.drop(544765)
    words = words.drop(430514)

    features = store_feat.select("words")
    features.set_index("Id", inplace=True, drop=False)

    if False:
        first = words.iloc[:20]
        plt.figure()
        plt.xlabel(r"Word")
        plt.ylabel(r"Percentage of occurrence")
        plt.bar(np.arange(1, first.shape[0] + 1), first.n.values * 100. / words.n.sum(),
                align="center", color="k", alpha=0.6)
        plt.xticks(np.arange(1, first.shape[0] + 1), first.words.values, rotation=90)
        plt.savefig("./plots/mostcommonwords.pdf")

    if False:  # did the number of answers change, e.g. for different time periods
        plt.figure()
        plt.xlabel(r"Number of answers to a question")
        plt.ylabel("Normalized counts (a.u.)")
        qs.AnswerCount.hist(range=[-0.5, 29.5], bins=30, color="r", normed=True, histtype="step", label="2017", axes=plt.gca())
        plt.legend(loc="best")
        plt.savefig("./plots/nanswers_time.pdf")

    # some datetime conv
    datecols = ["CreationDate"]
    for datecol in datecols:
        qs[datecol] = pd.to_datetime(qs[datecol], origin="julian", unit="D")

    # join in information about occurring words, probabilities etc
    qs = qs.join(features, how="inner", rsuffix="_r")
    qs.head()

    # transforming tags
    qs["Tags"] = qs.Tags.apply(lambda x: x.split(";")[1:])
    qs["hasAnswers"] = qs.AnswerCount > 1

    now = pd.datetime.now()
    qs["dt_created"] = now - qs.CreationDate

    if False:
        plt.figure()
        plt.xlabel(r"Number of replies to a question")
        plt.ylabel(r"Counts")
        qs.AnswerCount.hist(bins=100, range=(0, 10), color="k", alpha=0.8)
        plt.savefig("./plots/naswers_hist.pdf")

    if False:
        plt.figure()
        plt.xlabel(r"Days since a question was created")
        plt.ylabel(r"Number of answers")
        QuickSlicePlot(qs.dt_created.dt.days, qs.AnswerCount, qs.Score, zbins=1, xbins=10,
                       yrange=[0, 3], axes=plt.gca(), outliers=False)
        plt.savefig("./plots/nanswers_vs_dt.pdf")

    qs = qs[qs.nwords > 5]

    # shuffling the df to avoid time ordering dependencies for now
    qs = shuffle(qs)
    print len(qs)

    # join information about first answer into the frame
    qs = qs.merge(answers[["ParentId", "CreationDate"]], how="left", left_on="Id", right_on="ParentId", suffixes=("", "_first"))

    # join in information about accepted answer
    qs = qs.merge(answers[["Id", "CreationDate"]], how="left", left_on="AcceptedAnswerId", right_on="Id", suffixes=("", "_acc"))

    qs["CreationDate_first"] = pd.to_datetime(qs.CreationDate_first, origin="julian", unit="D")
    qs["CreationDate_acc"] = pd.to_datetime(qs.CreationDate_acc, origin="julian", unit="D")

    # time between questions posing and first answer
    dtanswer = qs.CreationDate_first - qs.CreationDate
    dtanswer_acc = qs.CreationDate_acc - qs.CreationDate

    if False:
        from pyik.fit import ChiSquareFunction
        from pyik.numpyext import centers

        dthours = dtanswer.dt.total_seconds() * 1. / 60 / 60
        # dtcont = np.linspace(0, 14 * 24, 1000)

        counts, e = np.histogram(dthours, range=(0, 24 * 14), bins=24 * 14)
        # cens = centers(e)[0]

        # print counts, cens

        # fitmodel = lambda dt, pars: 10 ** (pars[0] - dt * pars[1]) + 10 ** (pars[2] - dt * pars[3])
        # fitmodel = lambda dt, pars: np.e ** pars[0] * dt ** pars[1]
        # chi2fct = ChiSquareFunction(fitmodel, counts, cens, yerrs=np.sqrt(counts))
        # pars, cov, chi2, ndof = chi2fct.Minimize(np.asfarray([2, 0.1, 5.3, 0.13]),
        #                                         lower_bounds=np.asfarray([1, 0.01, 5, 0.05]),
        #                                         upper_bounds=np.asfarray([3, 0.2, 6, 0.3]),
        #                                         method="PRAXIS")

        plt.figure(figsize=(8, 6))
        plt.xlabel(r"Time between question and answer in hours")
        plt.ylabel(r"Counts")
        plt.hist(dthours, bins=24 * 14,
                 color="k", alpha=0.8, range=(0, 14 * 24),
                 histtype="step", lw=2)

        # plt.plot(dtcont, fitmodel(dtcont, pars), "r-", lw=2)

        plt.semilogy(nonposy="clip")
        plt.savefig("./plots/dtanswer_hist.pdf")

    # modelling

    traincut = 300000
    qs = shuffle(qs)
    print len(qs)

    # e.g. for regression
    # labels = qs.AnswerCount

    # e.g. classification, i.e. logistic regression
    labels = qs.hasAnswers

    pipe_tags = Pipeline([
                         ("cst", ColumnSelectTransformer(["Tags"])),
                         ("dec", DictEncoder("Tags")),
                         ("dvec", DictVectorizer(sparse=True)),
                         ("tfid", TfidfTransformer()),
                         # ("poly", PolynomialFeatures(degree=2)),  # not working???
                         # ("ridge", Ridge(alpha=10.0))
                         ("logi", LogisticRegression())
                         # ("kridge", KernelRidge(alpha=1.))  # runs out of memory quickly while fitting...
                         # ("svr", SVR())
                         ])

    pipe_tags.fit(qs.iloc[:traincut], labels.iloc[:traincut])
    pred = pipe_tags.predict(qs.iloc[traincut:])
    print np.column_stack((pred, labels.iloc[traincut:]))
    print labels.iloc[traincut:].shape[0] * 1. / 2, labels.iloc[traincut:].shape[0]
    print np.sum(pred == labels.iloc[traincut:])
    print pipe_tags.score(qs.iloc[traincut:], labels.iloc[traincut:])

    # pipe_tags.named_steps["ridge"].get_params()

    qs.columns
    qs.hot_indices.head()

    pipe_words = Pipeline([
                          ("cst", ColumnSelectTransformer(["hot_indices"])),
                          ("dec", DictEncoder("hot_indices")),
                          ("dvec", DictVectorizer(sparse=True)),
                          ("tfid", TfidfTransformer()),
                          # ("ridge", Ridge(alpha=1.0))
                          ("logi", LogisticRegression())
                          ])

    pipe_words.fit(qs.iloc[:traincut], labels.iloc[:traincut])
    pred = pipe_words.predict(qs.iloc[traincut:])
    print np.column_stack((pred, labels.iloc[traincut:]))
    print labels.iloc[traincut:].shape[0] * 1. / 2, labels.iloc[traincut:].shape[0]
    print np.sum(pred == labels.iloc[traincut:])
    print pipe_words.score(qs.iloc[traincut:], labels.iloc[traincut:])

    cv = model_selection.ShuffleSplit(n_splits=20, test_size=0.2, random_state=42)
    def compute_error(est, X, y):
        return -model_selection.cross_val_score(est, X, y, cv=cv, scoring='neg_mean_squared_error').mean()

    qs["titlelen"] = qs["Title"].apply(len)

    qs.columns
    qs[["BodyNCodes", "BodyNQMarks", "BodySize", "nwords", "ratio",
        "prob_bern", "ordersum", "titlelen"]].head()

    cols = ["BodyNCodes", "BodyNQMarks", "BodySize", "nwords", "ratio", "prob_bern", "ordersum", "titlelen"]
    # normalize columns
    for col in cols:
        quants = qs[col].quantile([0.01, 0.99]).values
        print col, quants
        qs["%s_norm" % col] = (qs[col] - quants[0]) / (quants[1] - quants[0])

    qs[["BodyNCodes_norm", "BodyNQMarks_norm", "BodySize_norm", "nwords_norm", "ratio_norm",
        "prob_bern_norm", "ordersum_norm", "titlelen_norm"]].head()

    qs["ordersum_norm_trafo"] = 1. / qs.ordersum_norm
    qs["titlelen_norm_trafo"] = 1. / qs.titlelen_norm

    labels = qs.hasAnswers

    pipe_feat = Pipeline([
           # ("cst", ColumnSelectTransformer(["BodyNCodes_norm", "BodyNQMarks_norm", "BodySize_norm", "nwords_norm", "ratio_norm",
                                            # "prob_bern_norm", "ordersum_norm", "titlelen_norm"])),
             ("cst", ColumnSelectTransformer(["BodyNCodes_norm", "BodyNQMarks_norm", "nwords_norm", "BodySize_norm",
                                              "ratio_norm", "ordersum_norm", "titlelen_norm"])),
             ("poly", PolynomialFeatures(degree=2)),
           # ("ridge", Ridge(alpha=1.0))
             ("logi", LogisticRegression())
        ])

    pipe_feat.fit(qs.iloc[:traincut], labels.iloc[:traincut])
    pred = pipe_feat.predict(qs.iloc[traincut:])
    print np.column_stack((pred, labels.iloc[traincut:]))
    print labels.iloc[traincut:].shape[0] * 1. / 2, labels.iloc[traincut:].shape[0]
    print np.sum(pred == labels.iloc[traincut:])
    print pipe_feat.score(qs.iloc[traincut:], labels.iloc[traincut:])
    print pipe_feat.predict_proba(qs.iloc[traincut:])

    # print compute_error(pipe_feat, qs, qs.AnswerCount)

    labels = qs.hasAnswers

    pipe_prob = Pipeline([
                         ("cst", ColumnSelectTransformer(["prob_bern_norm"])),
                         ("logi", LogisticRegression())
                         ])

    pipe_prob.fit(qs.iloc[:traincut], labels.iloc[:traincut])
    pred = pipe_prob.predict(qs.iloc[traincut:])
    print np.column_stack((pred, labels.iloc[traincut:]))
    print labels.iloc[traincut:].shape[0] * 1. / 2, labels.iloc[traincut:].shape[0]
    print np.sum(pred == labels.iloc[traincut:]), np.sum(pred == labels.iloc[traincut:]) * 1. / labels.iloc[traincut:].shape[0]
    print pipe_feat.score(qs.iloc[traincut:], labels.iloc[traincut:])
    print pipe_feat.predict_proba(qs.iloc[traincut:])
    print pipe_feat.predict_proba(qs.iloc[traincut:]).shape

    # print compute_error(pipe_feat, qs, qs.AnswerCount)

    from sklearn.pipeline import FeatureUnion

    union = FeatureUnion([
                         ("tags", EstimatorTransformer(pipe_tags)),
                         ("words", EstimatorTransformer(pipe_words)),
                         ("feat", EstimatorTransformer(pipe_feat)),
                         ("prob", EstimatorTransformer(pipe_prob))
                         ])

    pipe_all = Pipeline([
                        ("union", union),
                        ("ridge", Ridge(alpha=3.0))
                        # ("comb", CustomCombiner())
                        ])

    pipe_all.fit(qs.iloc[:traincut], labels.iloc[:traincut])
    pred = np.around(pipe_all.predict(qs.iloc[traincut:]))
    print np.column_stack((pred, labels.iloc[traincut:]))
    print labels.iloc[traincut:].shape[0] * 1. / 2, labels.iloc[traincut:].shape[0]
    print np.sum(pred == labels.iloc[traincut:]), np.sum(pred == labels.iloc[traincut:]) * 1. / labels.iloc[traincut:].shape[0]
    print pipe_feat.score(qs.iloc[traincut:], labels.iloc[traincut:])
    # print pipe_feat.predict_proba(qs.iloc[traincut:])


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("Stackoverflow word analysis steered by config file.")
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

    data = PrepareData(cfg)
    # SelectionAndShuffling(cfg)
    # SimpleAnalysis(cfg)
