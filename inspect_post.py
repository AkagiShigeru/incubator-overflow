#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Inspect a random or fixed post from stackoverflow dump
#
#
import os
from glob import glob
import numpy as np
import gzip

files = glob("/home/alex/data/stackexchange/overflow/posts/*.txt.gz")

print "Number of posts in directory:", len(files)

# rani = 234344
rani = np.random.randint(len(files))

with gzip.open(files[rani]) as gf:
    print gf.read()
