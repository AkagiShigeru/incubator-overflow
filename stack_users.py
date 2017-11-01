#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Tools to read and convert stackoverflow user info xml files into hdf files
#
#
import os
import gzip

from collections import defaultdict

from IPython import embed

from stack_util import *


all_cols_and_defaults = {"AboutMe": "", "AccountId": -1, "Age": -1,
                         "CreationDate": -1, "DisplayName": "", "DownVotes": -1,
                         "Id": -1, "LastAccessDate": -1, "Location": "",
                         "ProfileImageUrl": "", "Reputation": -1, "UpVotes": -1,
                         "Views": -1, "WebsiteUrl": ""}

cols_and_defaults = {"AboutMe": "", "AccountId": -1, "Age": -1,
                     "CreationDate": -1, "DisplayName": "", "DownVotes": -1,
                     "Id": -1, "LastAccessDate": -1, "Location": "",
                     "Reputation": -1, "UpVotes": -1, "Views": -1, "WebsiteUrl": ""}

cols_and_dtypes = {"AboutMe": str, "AccountId": int, "Age": int,
                   "CreationDate": int, "DisplayName": str, "DownVotes": int,
                   "Id": int, "LastAccessDate": int, "Location": str,
                   "ProfileImageUrl": str, "Reputation": int, "UpVotes": int,
                   "Views": int, "WebsiteUrl": str}

min_sizes = {"AboutMe": 100, "WebsiteUrl": 30, "DisplayName": 30, "Location": 30}

# user-defined columns with derived quantities
own_cols = {}

data_cols = ["AccountId", "Age", "CreationDate",
             "DownVotes", "Id", "LastAccessDate",
             "Location", "Reputation", "UpVotes", "Views"]

cols_and_converters = {"AboutMe": lambda t: t[:100],
                       "CreationDate": ConvertToJulianDate, "DisplayName": lambda t: t[:30],
                       "LastAccessDate": ConvertToJulianDate, "Location": lambda t: t[:30],
                       "WebsiteUrl": lambda t: t[:30]}


def CreateUserHDFStore(finp, outstore):

    user_dict = defaultdict(list)

    store = pd.HDFStore(outstore, "w", complib="blosc", complevel=9)
    store.put("users", pd.DataFrame(), format="table", data_columns=data_cols,
              min_itemsize=min_sizes)

    n = 0

    for entry in IterateZippedXML(finp):

        n += 1

        entrydict = dict(ParsePostFromXML(entry))

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
                        AssertionError("Couldn't convert %s to proper expected dtype!" % val)

                user_dict[ename].append(val)
            else:
                user_dict[ename].append(edefault)

        for entry in entrydict:
            assert entry in all_cols_and_defaults, "Unexpected entry with key: %s" % entry

        if n % 100000 == 0:

            print "Processed %i users!" % n

            df = pd.DataFrame(user_dict)
            store.append("users", df, format="table",
                         data_columns=data_cols, min_itemsize=min_sizes)
            user_dict.clear()
            del df

    # we need to push the remainder of users left in the dictionary
    if user_dict.values() != []:

        df = pd.DataFrame(user_dict)
        store.append("users", df, format="table",
                     data_columns=data_cols, min_itemsize=min_sizes)
        user_dict.clear()
        del df

    # ...and always close the store ;)
    store.close()

    return True


if __name__ == "__main__":
    fuser = "/home/alex/data/stackexchange/overflow/stackoverflow.com-Users.7z"

    CreateUserHDFStore(fuser, "/home/alex/data/stackexchange/overflow/caches/users.hdf5")
