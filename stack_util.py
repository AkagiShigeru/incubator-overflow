#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Simple utility functions used within the other scripts
#
#
import re
import pandas as pd
import numpy as np

from scipy.stats.mstats import mquantiles

from pyik.performance import pmap, cached

from lxml.html import fromstring
from HTMLParser import HTMLParser

from libarchive.public import file_reader


g_carr = ["k", "purple", "green", "#b22222", "#4682b4", "m", "r", "b", "c",
          "brown", "grey", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c",
          "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a",
          "#040073", "#D500D2", "#196A00", "#B89A79", "#900000", "#5A639F"] * 9
g_ls = ["-", "--", "-.", ":", "-"] * 10
g_markers = ["o", "s", "^", "H", "v", "<", ">", "D", "*", "p",
             "1", "2", "3", "4", "_", "d", "h", "x", "|", "."] * 10

cols_desc = {"AnswerCount": "Number of replies", "BodyNCodes": "Number of code tags",
             "BodyNQMarks": "Number of question marks", "BodySize": "Length of post text",
             "titlelen": "Length of title", "nwords": "Number of meaningful words",
             "ordersum": "Word prevalence (sum)", "ordermean": "Word prevalence (average)",
             "orderstd": "Word prevalence (std)", "ratio": "Ratio of number of verbs to nouns",
             "Score": "Question score", "prob_bern": "Joined bernoulli probability of words"}

hparser = HTMLParser()


def YieldDBPosts(conn, nchunk=1000):
    cursor = conn.execute("SELECT post FROM posts")
    while True:
        results = cursor.fetchmany(nchunk)
        if not results:
            break
        for result in results:
            yield result


def GetDBPosts(idlist, conn):
    posts = []
    for idi in idlist:
        posts.append(conn.execute("SELECT post FROM posts WHERE id=?", (idi,)).fetchall()[0][0])
    return posts


def local_import(x):
    """Only works on Unix and x may not be the name of a builtin module"""
    import os
    import sys

    path, fileName = os.path.split(x)
    if not path:
        path = "."
    base, ext = os.path.splitext(fileName)
    save = sys.dont_write_bytecode
    save_path = list(sys.path)
    sys.dont_write_bytecode = True
    sys.path = [path]
    rv = __import__(base)
    sys.dont_write_bytecode = save
    sys.path = save_path
    sys.path.pop()
    return rv


def SelectUniformlyFromColumn(df, col, n=100000, randomize=True):
    from sklearn.utils import shuffle
    uniques = df[col].unique()
    nu = n // len(uniques)
    navailable = df.groupby(col).apply(len)
    print "Available overall grouped counts:", navailable
    if min(navailable) < nu:
        print "Warning! There are not enough unique items of each category to sample, \
               output df will have reduced size of %i per group!" % min(navailable)
        nu = min(navailable)
    new = None
    for unique in uniques:
        sel = df[df[col] == unique].sample(nu)
        if new is None:
            new = sel
        else:
            new = new.append(sel)
    print "Grouped counts after selection:", new.groupby(col).apply(len)
    if randomize:
        return shuffle(new)
    else:
        return new


def CleanString(s):
    return fromstring(s).text_content().encode("ascii", "ignore")


def UnescapeHTML(s):
    return hparser.unescape(s)


def ParsePostFromXML(l):
    return re.findall(r"\s?([A-Z][A-Za-z]+)=\"([^\"]*)\"\s?", l)


def CleanTags(tagstring):
    return tagstring.replace("&gt;", "").replace("&lt;", ";")


def ConvertToJulianDate(datestr):
    return pd.to_datetime(datestr).to_julian_date()


def SplitTags(tagstring):
    """ Splits according to html entities of <> which bundle tags in the xml files."""
    cleaned = tagstring.replace("&gt;", "")
    if cleaned != "":
        split = cleaned.split("&lt;")
        if len(split) > 0:
            return split[1:]
    return []


def IterateZippedXML(zf, delim=" />\r\n  <row", debug=False):
    """ Generator that returns lines of zipped file line by line."""
    with file_reader(zf) as zhandler:
        for onefile in zhandler:
            line = ""
            for block in onefile.get_blocks():
                line += block
                if debug:
                    print block
                while delim in line:
                    pos = line.index(delim)
                    yield line[:pos].strip() + "/>"
                    line = line[pos + len(delim):].strip()
                if delim in line:
                    yield line + "/>"


def UncFromCov(cov):
  cov = np.atleast_1d(cov)
  assert cov.shape[0] == cov.shape[1], "Covariance matrix has wrong shape!"
  return np.sqrt([cov[i][i] for i in range(len(cov))])


def CovarianceIsLegit(cov):
  for i in range(len(cov)):
    if cov[i][i] <= 0.:
      return False
  return True


def QuickSlicePlot(x, y, z, xbins=0, nxbins=None, zbins=5, zrange=None,
                   xlabel=None, ylabel=None, zlabel=None, xRange=None, xmean=False,
                   yrange=[-1., 1.], zdict=None, ztrafo=None, zdigs=None,
                   draw="amv", outliers=True, color=None, marker=None,
                   label=None, labelloc="best", visshift=None, offsetY=0,
                   shiftfirst=False, fit_model=None, fit_starts=None,
                   textpos=None, axes=None, **kwargs):
    """
    Function to quickly produce plots of y(x) for different slices in z.

    Use case: interactively inspecting new data
    Creates a new figure and axis in case axes is None.
    """
    from matplotlib import pyplot as plt
    import numpy as np
    from pyik.mplext import ViolinPlot
    from scipy.stats.mstats import mquantiles
    from pyik.fit import ChiSquareFunction
    from pyik.numpyext import bin

    if axes is None:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    else:
        plt.sca(axes)
        ax = axes

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    # allows for transformation of z arguments in labels
    if ztrafo is None:
        ztrafo = lambda x: x

    iszunique = False

    if zbins == "unique":

        try:

            # scale of uniqueness is hardcoded! could be optimized
            z = np.around(z, 4)
            zuniq = np.unique(z)
            zuniq.sort()

        except:

            zuniq = np.unique(z)

        bindata = []

        for iz in zuniq:

            xsel = x[z == iz]
            ysel = y[z == iz]

            bindata.append(np.column_stack((xsel, ysel)))

        iszunique = True

    else:

        bindata, zedgs = bin(z, zip(x, y), bins=zbins, range=zrange)

    ic = 0

    visshifth = 0.
    if shiftfirst and visshift is not None:
        visshifth = visshift

    for i in range(len(bindata)):

        datai = np.asfarray(bindata[i])

        if len(datai) > 0:

            if xRange is not None:

                datai = datai[(datai.T[0] >= min(xRange)) &
                              (datai.T[0] < max(xRange))]

            if xbins is None:  # unique x values

                xuniq = np.unique(np.around(datai.T[0], 5))
                yvals = []

                for xone in xuniq:

                    dataii = datai[np.absolute(datai.T[0] - xone) < 1e-4]
                    yvals.append(dataii.T[1])

                xs = xuniq
                ys = yvals
                bins = None

            elif type(xbins) is int and xbins == 0:  # optimize number of bins

                if nxbins is None:
                    nbins = max(5, int(np.ceil(len(datai) * 1. / 30.)))
                else:
                    nbins = nxbins

                bins = mquantiles(datai.T[0], np.linspace(0., 1., nbins + 1))
                xs = datai.T[0]
                ys = datai.T[1]

            else:

                bins = xbins
                xs = datai.T[0]
                ys = datai.T[1]

            if iszunique:
                if zdict is not None:
                    lab = r"%s" % zdict[zuniq[i]]
                else:
                    if zdigs is not None:
                        lab = (r"$ %%.%if $" % zdigs) % ztrafo(zuniq[i])
                    else:
                        try:

                            ldigs = min(
                                max(2 - np.floor(np.abs(np.log10(zuniq[i]))), 0), 2)

                            if np.isfinite(ldigs):
                                ldigs = int(ldigs)
                            else:
                                ldigs = 0

                            lab = (r"$ %%.%if $" % ldigs) % ztrafo(zuniq[i])

                        except:
                            lab = r"%s" % ztrafo(zuniq[i])
            else:

                ldigs = min(
                    max(2 - np.floor(np.abs(np.log10(ztrafo(zedgs[i])))), 0), 2)

                if np.isfinite(ldigs):
                    ldigs = int(ldigs)
                else:
                    ldigs = 0

                if zdigs is not None:
                    ldigs = zdigs

                lab = (r"$ [ %%.%if, %%.%if ] $" % (ldigs, ldigs)) % (
                    ztrafo(zedgs[i]), ztrafo(zedgs[i + 1]))

            if zlabel is None:
                lab = None

            if label is not None:
                lab = label

            xcens, xhws, ymeans, ystds, ymeds, ymads, ns = ViolinPlot(xs, ys, bins=bins,
                                                                      color=g_carr[
                                                                          ic] if color is None else color,
                                                                      marker=g_markers[
                                                                          ic] if marker is None else marker,
                                                                      offsetX=visshifth, offsetY=offsetY, textpos=textpos,
                                                                      draw=draw, outliers=outliers, xmean=xmean, axes=ax, label=lab,
                                                                      **kwargs)

            if fit_model is not None and fit_starts is not None:

                print "Fitting profiles to model with {0} parameters!".format(len(fit_starts))

                try:

                    chi2fct = ChiSquareFunction(
                        fit_model, xcens, ymeans, yerrs=ystds / np.sqrt(ns))

                    fitpars, fitcov, fitchi2, fitndof = chi2fct.Minimize(fit_starts,
                                                                         lower_bounds=None,
                                                                         upper_bounds=None,
                                                                         method="PRAXIS",
                                                                         absolute_tolerance=1e-12,
                                                                         covarianceMethod="slow")
                    print fitchi2 / fitndof
                    print fitpars, UncFromCov(fitcov)

                    xcont = np.linspace(
                        (xcens - xhws)[0], (xcens + xhws)[-1], 100)

                    plt.plot(xcont, fit_model(xcont, fitpars), ls="-",
                             color=g_carr[ic] if color is None else color)

                except:
                    print "Fit failed! Continuing!"

            if visshift is not None:
                visshifth += visshift

            ic += 1

    if zlabel is not None:
        plt.legend(loc=labelloc, ncol=1 + len(bindata) // 4, title=zlabel)
    else:
        if label is not None and axes is None:
            plt.legend(loc="best")

    plt.xlim(min(x), max(x))

    if yrange is not None:
        plt.ylim(yrange[0], yrange[-1])

    if axes is None:
        plt.show()
        return xcens, xhws, ymeans, ystds, ymeds, ymads, ns
    else:
        return xcens, xhws, ymeans, ystds, ymeds, ymads, ns
