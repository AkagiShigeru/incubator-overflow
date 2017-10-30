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
from util_general import QuickSlicePlot

from stack_util import *


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
