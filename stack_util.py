#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Simple utility functions used within the other scripts
#
#
import re

from lxml.html import fromstring
from HTMLParser import HTMLParser


hparser = HTMLParser()


def CleanString(s):
    return fromstring(s).text_content().encode("ascii", "ignore")


def UnescapeHTML(s):
    return hparser.unescape(s)


def ParsePostFromXML(l):
    return re.findall(r"\s?([A-Z][A-Za-z]+)=\"([^\"]*)\"\s?", l)


def CleanTags(tagstring):
    return tagstring.replace("&gt;", "").replace("&lt;", ";")


def SplitTags(tagstring):
    """ Splits according to html entities of <> which bundle tags in the xml files."""
    cleaned = tagstring.replace("&gt;", "")
    if cleaned != "":
        split = cleaned.split("&lt;")
        if len(split) > 0:
            return split[1:]
    return []
