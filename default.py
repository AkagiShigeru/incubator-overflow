#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Default config file for stack words analysis.
#  For a custom analysis, make a copy of this default config file and work from there.
#
#
import os
from glob import glob


paths = {}
paths["caches"] = "/home/alex/data/stackexchange/overflow/caches/"
paths["db"] = os.path.join(paths["caches"], "posts.db")
paths["metas"] = paths["caches"]
paths["features"] = os.path.join(paths["caches"], "features/")
paths["dictionaries"] = os.path.join(paths["caches"], "dictionaries/")
