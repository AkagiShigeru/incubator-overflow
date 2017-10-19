#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Inspect a random or fixed post from stackoverflow dump
#
#
import os
import re
from glob import glob
import numpy as np
import pandas as pd
import gzip

files = glob("/home/alex/data/stackexchange/overflow/posts/*.txt.gz")
hdfpath = "/home/alex/data/stackexchange/overflow/caches/posts_all.hdf5"

print "Number of posts in directory:", len(files)

# rani = 1798777
rani = np.random.randint(len(files))

with gzip.open(files[rani]) as gf:
    print ">>>"
    print gf.read()
    print "<<<"

store = pd.HDFStore(hdfpath, "r", complib="blosc", complevel=9)

cols = None
smask = store.select_as_coordinates("posts", "Id == %i" % int(re.findall(r"\d+", os.path.split(files[rani])[-1])[0]))
posts = store.select("posts", where=smask)

print "Information about this post:\n"
print posts.iloc[0]
