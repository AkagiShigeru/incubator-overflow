#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Default config file for stack words analysis.
#  For a custom analysis, make a copy of this default config file and work from there.
#
#
import os
from copy import copy
import sqlite3


paths = {}
paths["caches"] = "/home/alex/data/stackexchange/overflow/caches/"
paths["db"] = os.path.join(paths["caches"], "posts.db")
paths["metas"] = paths["caches"]
paths["features"] = os.path.join(paths["caches"], "features_new/")
paths["dictionaries"] = os.path.join(paths["caches"], "dictionaries/")


data = {}

posts_path = os.path.join(paths["db"])
meta_path = os.path.join(paths["metas"], "posts_2017.hdf5")
dict_path = os.path.join(paths["dictionaries"], "words_2017.hdf5")
features_path = os.path.join(paths["features"], "features_2017.hdf5")

dbonn = sqlite3.connect(posts_path)


# for future redesign (possibly)
# data = []

# meta = {}
# meta["Id"] = "meta_2017"
# meta["path"] = os.path.join(paths["metas"], "posts_2017.hdf5")
# data.append(meta.copy())
