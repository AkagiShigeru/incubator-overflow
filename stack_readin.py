#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Tools to read and convert stackoverflow xml files into hdf files
#
#
import os
import gzip

from collections import defaultdict

from IPython import embed

from stack_util import *


all_cols_and_defaults = {"AcceptedAnswerId": -1, "AnswerCount": -1, "Body": "", "Body_unesc": "",
                         "ClosedDate": -1., "CommentCount": -1, "CommunityOwnedDate": -1., "CreationDate": -1.,
                         "FavoriteCount": -1, "Id": -1, "LastActivityDate": -1., "LastEditDate": -1.,
                         "LastEditorDisplayName": "", "LastEditorUserId": -1, "OwnerDisplayName": "",
                         "OwnerUserId": -1, "ParentId": -1, "PostTypeId": -1, "Score": -1,
                         "Tags": "", "Title": "", "ViewCount": -1}

cols_and_defaults = {"AcceptedAnswerId": -1, "AnswerCount": -1, "ClosedDate": -1.,
                     "CommentCount": -1, "CommunityOwnedDate": -1., "CreationDate": -1.,
                     "FavoriteCount": -1, "Id": -1, "LastActivityDate": -1.,
                     "LastEditDate": -1., "LastEditorUserId": -1,
                     "OwnerUserId": -1, "ParentId": -1, "PostTypeId": -1, "Score": -1,
                     "Tags": "", "Title": "", "ViewCount": -1}

cols_and_dtypes = {"AcceptedAnswerId": int, "AnswerCount": int, "Body": str, "ClosedDate": float,
                   "CommentCount": int, "CommunityOwnedDate": float, "CreationDate": float,
                   "FavoriteCount": int, "Id": int, "LastActivityDate": float, "LastEditDate": float,
                   "LastEditorDisplayName": str, "LastEditorUserId": int, "OwnerDisplayName": str,
                   "OwnerUserId": int, "ParentId": int, "PostTypeId": int, "Score": int,
                   "Tags": str, "Title": str, "ViewCount": int}

min_sizes = {"Body": 2000, "Title": 300, "Tags": 150}

# user-defined columns with derived quantities
own_cols = {"BodySize": lambda e: len(e["Body_unesc"]),
            "BodyNQMarks": lambda e: e["Body_unesc"].count("?"),
            "BodyNCodes": lambda e: e["Body_unesc"].count("<code>")}

data_cols = ["AnswerCount", "CreationDate", "CommentCount", "FavoriteCount", "Id", "Score",
             "PostTypeId", "ViewCount", "Tags"]

cols_and_converters = {"Tags": lambda t: CleanTags(t)[:150], "Body": lambda x: "",
                       "Title": lambda t: t[:300], "OwnerDisplayName": lambda n: n[:30],
                       "LastEditorDisplayName": lambda n: n[:30],
                       "CreationDate": ConvertToJulianDate, "ClosedDate": ConvertToJulianDate,
                       "CommunityOwnedDate": ConvertToJulianDate, "LastActivityDate": ConvertToJulianDate,
                       "LastEditDate": ConvertToJulianDate}


# TODO: function to read split and unpacked XML in parallel (using pmap)
def ReadInParallel(xmls):
    pass


def DumpIntoSQLite(finp, outpath):

    import sqlite3
    conn = sqlite3.connect(outpath)

    # conn.execute("DROP TABLE posts")
    conn.execute("CREATE TABLE posts (id int, post text)")

    n = 0
    for post in IterateZippedXML(finp):

        entrydict = dict(ParsePostFromXML(post))

        body_esc = UnescapeHTML(entrydict["Body"].decode("utf-8"))
        oneid = int(entrydict["Id"])

        conn.execute("INSERT INTO posts VALUES (?, ?)", (oneid, body_esc))

        if n % 200000 == 0:
            conn.commit()

        n += 1

        print "Processed %i posts." % n

    conn.commit()

    conn.close()


def CreateHDFStores(finp, outstore, fields=None, dump_posts=False, limit=None,
                    create_word_dicts=True, year=2008):
    """ Uses other utility function to create hdf store with DataFrame of posts.
        Optionally dumps individual posts into zipped files on the hard disk."""
    base = os.path.split(finp)[0]
    print "Base path:", base

    base_out = os.path.splitext(outstore)[0]

    post_dict = defaultdict(list)

    word_dict = defaultdict(int)
    word_dict_python = defaultdict(int)
    word_dict_cpp = defaultdict(int)

    # store = None
    # year = None

    if fields is None:
        fields = cols_and_defaults

    isizes = {k: min_sizes[k] for k in fields.keys() if k in min_sizes}

    store = pd.HDFStore("%s%i.hdf5" % (base_out, year), "w", complib="blosc", complevel=9)
    store.put("posts", pd.DataFrame(), format="table",
              data_columns=data_cols, min_itemsize=isizes)

    n = 0
    for post in IterateZippedXML(finp):

        n += 1

        # read-in limit reached
        if limit and n > limit:
            break

        entrydict = dict(ParsePostFromXML(post))

        post_year = pd.to_datetime(entrydict["CreationDate"]).year

        # post are sorted with a few exceptions at the transition period between years
        if post_year > year + 1:
            break

        if post_year != year:
            continue

        entrydict["Body_unesc"] = UnescapeHTML(entrydict["Body"].decode("utf-8"))

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

        if create_word_dicts:
            p = entrydict["Body_unesc"]
            p_words = re.sub("<.*?>", "", p).replace(",", " ").replace("\n", " ").split()
            for word in p_words:
                word_dict[word] += 1

            if "Tags" in entrydict:
                t = entrydict["Tags"]
                if "python" in t:
                    for word in p_words:
                        word_dict_python[word] += 1
                if "c++" in t:
                    for word in p_words:
                        word_dict_cpp[word] += 1

        # for old non-parallel method
        # if year is None or year != post_year:

        #     print "Starting store for events in year %i." % post_year

        #     # dump events that are still from last year into corresponding store
        #     if store is not None:
        #         df = pd.DataFrame(post_dict)
        #         store.append("posts", df, format="table",
        #                      data_columns=data_cols, min_itemsize=isizes)
        #         post_dict.clear()
        #         del df
        #         store.close()

        #     store = pd.HDFStore("%s%i.hdf5" % (base_out, post_year), "w", complib="blosc", complevel=9)
        #     if "posts" not in store:
        #         store.put("posts", pd.DataFrame(), format="table",
        #                   data_columns=data_cols, min_itemsize=isizes)
        #     year = post_year

        for ename, edefault in fields.items():

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
                        AssertionError("Couldn't convert %s to proper expected dtype!" % val)

                post_dict[ename].append(val)
            else:
                post_dict[ename].append(edefault)

        for entry in entrydict:
            assert entry in all_cols_and_defaults, "Unexpected entry with key: %s" % entry

        # calculate and append custom entries
        for custom in own_cols:
            post_dict[custom].append(own_cols[custom](entrydict))

        if n % 100000 == 0:

            print "Processed %i posts!" % n

            df = pd.DataFrame(post_dict)
            store.append("posts", df, format="table",
                         data_columns=data_cols, min_itemsize=isizes)
            post_dict.clear()
            del df

    # we need to push the remainder of posts left in the dictionary
    if post_dict.values() != []:

        df = pd.DataFrame(post_dict)
        store.append("posts", df, format="table",
                     data_columns=data_cols, min_itemsize=isizes)
        post_dict.clear()
        del df

    # ...and always close the store ;)
    store.close()

    # embed()

    return True


if __name__ == "__main__":
    f = "/home/alex/data/stackexchange/overflow/stackoverflow.com-Posts.7z"

    # CreateHDFStores(f, "/home/alex/data/stackexchange/overflow/caches/posts_first5M.hdf5",
    #                 limit=5000000)

    # CreateHDFStores(f, "/home/alex/data/stackexchange/overflow/caches/posts_all.hdf5",
    #                 dump_posts=True)

    def CreateYearly(y):
        CreateHDFStores(f, "/home/alex/data/stackexchange/overflow/caches/posts_.hdf5",
                        dump_posts=False, create_word_dicts=False, year=y)

    from pyik.performance import pmap
    import numpy as np
    # years = np.arange(2008, 2018)
    years = [2009, 2011]
    pmap(CreateYearly, years, numprocesses=8)

    # DumpIntoSQLite(f, "/home/alex/data/stackexchange/overflow/caches/posts.db")

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
