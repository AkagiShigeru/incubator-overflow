#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Tools to analyze posts (e.g. user input) based on fitted models and processing routines
#
#
import os
from stack_nlp import *
from stack_util import local_import


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

    userpost = {"post": "This is crappy testpost",
                "title": "testpost",
                "date": "26/11/2017",
                "user": "testuser"}

    AnalyzePost(cfg, userpost=userpost, pid=None)
