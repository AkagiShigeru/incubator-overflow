#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#
#  Tools to read and convert stackoverflow xml files
#
#
from collections import defaultdict
import pandas as pd

from libarchive.public import file_reader

from xml.etree import cElementTree

from IPython import embed

f = "/home/alex/data/stackexchange/stackoverflow.com-Posts.7z"


def ParseZippedXML(zf):
    date_counts = defaultdict(int)
    with file_reader(f) as e:
        for entry in e:
            iterparser = cElementTree.iterparse(entry, events=("start", ))
            _, root = iterparser.next()
            for _, element in iterparser:
                if element.tag == "row":
                    date_str = element.get("CreationDate", "").split("T")[0]
                    date_counts[date_str] += 1
                root.clear()
    return date_counts


if __name__ == "__main__":
    ParseZippedXML(f)
    embed()
