#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Tools to build word dictionaries from stack overflow posts
#
#
import os
from glob import glob

from collections import defaultdict

from IPython import embed

from stack_util import *
import sqlite3

from spacy.en import STOP_WORDS
import spacy

nlp = spacy.load("en_core_web_md")


def BuildDictionariesFromFile(finp, outp="./words.hdf5", limit=100000):

    base = os.path.split(finp)[0]
    print "Base path:", base

    word_dict = defaultdict(int)

    n = 0
    for post in IterateZippedXML(finp):

        n += 1

        # read-in limit reached
        if limit and n > limit:
            break

        entrydict = dict(ParsePostFromXML(post))

        entrydict["Body_unesc"] = UnescapeHTML(entrydict["Body"].decode("utf-8"))

        p = entrydict["Body_unesc"]

        p_clean = p.replace("\n", " ")

        # not greedy code removal
        p_clean = re.sub(r"<code>.*?</code>", " ", p_clean)
        p_clean = re.sub(r"<\/?\w*>", " ", p_clean)
        p_clean = p_clean.replace(":", "").lower()

        # print p_clean

        # for word in p_clean.split():
        for word in nlp(p_clean):
            if word.lemma_ not in STOP_WORDS:
                if word.pos_ in ["NOUN", "VERB", "ADV", "ADJ"]:
                    # print word, word.lemma_, word.tag_, word.pos_
                    word_dict[word.lemma_] += 1

        if n % 5000 == 0:
            print n, len(word_dict.keys())

    # saving
    words = pd.DataFrame({"words": word_dict.keys(), "n": word_dict.values()})
    store = pd.HDFStore(outp, "w", complib="blosc", complevel=9)
    store.put("all", words)
    store.close()

    return True


def BuildWordLists(finp, outstore, wdict=None, limit=1000000):

    base = os.path.split(finp)[0]
    print "Base path:", base

    assert wdict is not None, "Word dictionary must be specified for this function to work!"

    words = defaultdict(list)

    store = pd.HDFStore(outstore, "w", complib="blosc", complevel=9)
    store.put("words", pd.DataFrame(), format="table",
              data_columns=["Id"])

    n = 0
    for post in IterateZippedXML(finp):

        n += 1

        # read-in limit reached
        if limit and n > limit:
            break

        entrydict = dict(ParsePostFromXML(post))

        entrydict["Body_unesc"] = UnescapeHTML(entrydict["Body"].decode("utf-8"))

        # not greedy code removal
        p_clean = re.sub(r"<code>.*?</code>", " ", p_clean)
        p_clean = re.sub(r"<\/?\w*>", " ", p_clean)
        p_clean = p_clean.replace(":", "").lower()

        # print p_clean

        # for word in p_clean.split():
        for word in nlp(p_clean):
            if word.lemma_ not in STOP_WORDS:
                if word.pos_ in ["NOUN", "VERB", "ADV", "ADJ"]:
                    # print word, word.lemma_, word.tag_, word.pos_
                    word_dict[word.lemma_] += 1

        if n % 5000 == 0:
            print n, len(word_dict.keys())


# unfinished, not sure if makes sense due to very slow sqlite access...
def BuildDictionariesFromDB(meta_path, posts_path):

    print "Building dictionary for %s" % meta_path

    conn = sqlite3.connect(posts_path)

    store = pd.HDFStore(meta_path, "r", complib="blosc", complevel=9)

    chunks = store.select("posts", chunksize=10000, iterator=True)

    for chunk in chunks:

        for i in range(chunk.shape[0]):

            pid = chunk.iloc[i].Id

            print pid
            curr = conn.execute("SELECT post FROM posts WHERE id=?", (pid,))
            p = curr.fetchone()

            print p

            break

        break


if __name__ == "__main__":

    # hdf5 stores with post meta-data
    metas = sorted(glob("/home/alex/data/stackexchange/overflow/caches/posts_*.hdf5"))
    dbpath = "/home/alex/data/stackexchange/overflow/caches/posts.db"

    f = "/home/alex/data/stackexchange/overflow/stackoverflow.com-Posts.7z"

    # BuildDictionariesFromDB(metas[0], dbpath)
    # BuildDictionariesFromFile(f, limit=1000000)

    BuildWordLists(f, "./word_observed_.hdf5",
                   pd.HDFStore("words.hdf5", "r", complib="blosc", complevel=9).get("all"))
