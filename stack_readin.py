#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Tools to read and convert stackoverflow xml files into hdf files
#
#
import os
from libarchive.public import file_reader
import gzip

import pandas as pd

from collections import defaultdict

from IPython import embed

from stack_util import *


cols_and_defaults = {"AcceptedAnswerId": -1, "AnswerCount": -1, "Body": "", "ClosedDate": "",
                     "CommentCount": -1, "CommunityOwnedDate": "", "CreationDate": "",
                     "FavoriteCount": -1, "Id": -1, "LastActivityDate": "", "LastEditDate": "",
                     "LastEditorDisplayName": "", "LastEditorUserId": -1, "OwnerDisplayName": "",
                     "OwnerUserId": -1, "ParentId": -1, "PostTypeId": -1, "Score": -1,
                     "Tags": "", "Title": "", "ViewCount": -1}

cols_and_dtypes = {"AcceptedAnswerId": int, "AnswerCount": int, "Body": str, "ClosedDate": str,
                   "CommentCount": int, "CommunityOwnedDate": str, "CreationDate": str,
                   "FavoriteCount": int, "Id": int, "LastActivityDate": str, "LastEditDate": str,
                   "LastEditorDisplayName": str, "LastEditorUserId": int, "OwnerDisplayName": str,
                   "OwnerUserId": int, "ParentId": int, "PostTypeId": int, "Score": int,
                   "Tags": str, "Title": str, "ViewCount": int}

# user-defined columns with derived quantities
own_cols = {"BodySize": lambda e: len(e["Body"]),
            "BodyNQMarks": lambda e: e["Body"].count("?")}

cols_and_converters = {"Tags": lambda t: CleanTags(t)[:150], "Body": lambda x: "",
                       "Title": lambda t: t[:300], "OwnerDisplayName": lambda n: n[:30],
                       "LastEditorDisplayName": lambda n: n[:30]}


def IterateZippedXML(zf, delim=" />\r\n  <row", debug=False):
    """ Generator that returns lines of zipped file line by line."""
    with file_reader(zf) as zhandler:
        for onefile in zhandler:
            line = ""
            for block in onefile.get_blocks():
                line += block
                if debug:
                    print block
                while delim in line:
                    pos = line.index(delim)
                    yield line[:pos].strip() + "/>"
                    line = line[pos + len(delim):].strip()
                if delim in line:
                    yield line + "/>"


# TODO: function to read split and unpacked XML in parallel (using pmap)
def ReadInParallel(xmls):
    pass


def CreateHDFStores(finp, outstore, dump_posts=False, limit=None):
    """ Uses other utility function to create hdf store with DataFrame of posts.
        Optionally dumps individual posts into zipped files on the hard disk."""
    base = os.path.split(finp)[0]
    print "Base path:", base

    post_dict = defaultdict(list)

    store = pd.HDFStore(outstore, "w", complib="blosc", complevel=9)
    store.put("posts", pd.DataFrame(), format="table", data_columns=True)

    n = 0
    for post in IterateZippedXML(finp):

        n += 1

        # read-in limit reached
        if limit and n > limit:
            break

        entrydict = dict(ParsePostFromXML(post))

        # dump actual post entries into seperate files (not to blow up dataframe too much)
        if dump_posts:
            posttype = int(entrydict["PostTypeId"])
            answercount = 0
            if "AnswerCount" in entrydict:
                answercount = int(entrydict["AnswerCount"])

            # only saving questions (posttype == 1) with at least 3 answers and which are closed (properly responded to)
            if posttype == 1 and answercount > 2 and "ClosedDate" in entrydict:

                with gzip.open(os.path.join(base, "posts/post_%s.txt.gz" % entrydict["Id"]), "wb") as f:
                    try:
                        f.write(UnescapeHTML(entrydict["Body"]))
                    except:
                        f.write(entrydict["Body"])

        for ename, edefault in cols_and_defaults.items():

            if ename in entrydict:
                val = entrydict[ename]

                # optional conversion function
                if ename in cols_and_converters:
                    val = cols_and_converters[ename](val)

                # dtype conversion
                dtype = cols_and_dtypes[ename]
                if not isinstance(val, dtype):
                    try:
                        val = dtype(val)
                    except:
                        AssertionError("Couldn't convert %s to proper dtype!" % val)

                post_dict[ename].append(val)
            else:
                post_dict[ename].append(edefault)

        for entry in entrydict:
            assert entry in cols_and_defaults, "Unexpected entry with key: %s" % entry

        # calculate and append custom entries
        for custom in own_cols:
            post_dict[custom].append(own_cols[custom](entrydict))

        if n % 200000 == 0:

            print "Processed %i posts!" % n

            df = pd.DataFrame(post_dict)
            store.append("posts", df, format="table", data_columns=True,
                         min_itemsize={"Title": 300, "Tags": 150, "OwnerDisplayName": 30,
                                       "LastEditorDisplayName": 30})
            post_dict.clear()
            del df

    # we need to push the remainder of posts left in the dictionary
    if post_dict.values() != []:

        df = pd.DataFrame(post_dict)
        store.append("posts", df, format="table", data_columns=True)
        post_dict.clear()
        del df

    # ...and always close the store ;)
    store.close()

    return True


if __name__ == "__main__":
    f = "/home/alex/data/stackexchange/overflow/stackoverflow.com-Posts.7z"

    # CreateHDFStores(f, "/home/alex/data/stackexchange/overflow/caches/posts_first5M.hdf5",
    #                 limit=5000000)

    CreateHDFStores(f, "/home/alex/data/stackexchange/overflow/caches/posts_all.hdf5",
                    dump_posts=True)

    # for testing and debugging
    # i = 0
    # cols = defaultdict(int)
    # for l in IterateZippedXML(f):
    #     i += 1
    #     if i < 3:
    #         continue
    #     res = dict(ParsePostFromXML(l))
    #     if "Tags" in res:
    #         print res
    #         embed()
    #     for k in res.keys():
    #         cols[k] += 1
    #     if i == 1000000:
    #         break
