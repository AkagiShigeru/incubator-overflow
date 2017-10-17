#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Tools to read and convert stackoverflow xml files
#
from libarchive.public import file_reader

from collections import defaultdict
import pandas as pd
import re

from lxml.html import fromstring
# from xml.etree import cElementTree
from IPython import embed


def IterateZippedXML(zf, delim="\n", debug=False):
    """ Generator that returns lines of zipped file line by line."""
    with file_reader(zf) as zhandler:
        for onefile in zhandler:
            line = ""
            for block in onefile.get_blocks():
                line += block
                if debug:
                    print block
                while delim in line:
                    pos = line.index(delim) + 1
                    yield line[:pos].strip()
                    line = line[pos:].strip()
                yield line


def CleanString(s):
    return fromstring(s).text_content().encode("ascii", "ignore")


def InitializePostsParser():
    # format: <row Id="number" PostTypeId="number" CreationDate="2008-08-01T00:42:38.903" Score="number"
    #          ViewCount="number" Body="any text" OwnerUserId="number" LastEditorUserId="number"
    #          LastEditorDisplayName="name" LastEditDate="2016-12-20T19:22:26.510"
    #          LastActivityDate="2017-07-31T11:30:41.573"
    #          Title="title text" Tags="tag1;tag2;tag3"
    #          AnswerCount="number" CommentCount="number" FavoriteCount="number" />
    parser = re.compile(r"\s?([A-Z][A-Za-z]+)=\"(.*)\"\s?")
    return parser


def ParsePost(l, parser):
    # return [m.groupdict() for m in parser.finditer(l)]
    return re.findall(r"\s?([A-Z][A-Za-z]+)=\"([^\"]*)\"\s?", l)

# todo
# write out stuff in data frame

if __name__ == "__main__":
    f = "/home/alex/data/stackexchange/overflow/stackoverflow.com-Posts.7z"

    parser = InitializePostsParser()

    i = 0
    for l in IterateZippedXML(f):
        if i < 1:
            i += 1
            continue
        print l
        print CleanString(l)
        print ParsePost(l, parser)
        i += 1
        if i == 3:
            break

    # embed()
