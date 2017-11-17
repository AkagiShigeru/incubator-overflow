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
options["read"] = ["questions", "answers", "dictionary", "features"]
["questions", "answers", "dictionary", "features"]
