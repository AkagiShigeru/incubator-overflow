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


def GetRelevantWords(post, get_ratio=False):
    """
    Yields relevant words in post after some cleaning.
    Uses spacy to select words.
    """
    p_clean = post.replace("\n", " ")

    # non-greedy code removal
    p_clean = re.sub(r"<code>.*?</code>", " ", p_clean)
    p_clean = re.sub(r"<\/?\w*>", " ", p_clean)
    p_clean = p_clean.replace(":", "").lower()

    words = []

    n_noun = 0
    n_verb = 0

    for word in nlp(p_clean):
        if word.lemma_ not in STOP_WORDS.union(u"-PRON-"):
            if word.pos_ in ["NOUN", "VERB", "ADV", "ADJ"]:
                words.append(word.lemma_)
                if get_ratio:
                    if word.pos_ == "NOUN":
                        n_noun += 1
                    if word.pos_ == "VERB":
                        n_verb += 1

    if get_ratio:
        if n_noun > 0:
            return words, n_verb * 1. / n_noun
        else:
            return words, -1.
    else:
        return words


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

        for w in GetRelevantWords(entrydict["Body_unesc"]):
            word_dict[w] += 1

        if n % 5000 == 0:
            print n, len(word_dict.keys())

    # saving to hdf
    words = pd.DataFrame({"words": word_dict.keys(), "n": word_dict.values()})
    store = pd.HDFStore(outp, "w", complib="blosc", complevel=9)
    store.put("all", words)
    store.close()

    return True


def BuildWordLists(finp, outstore, wdict=None, limit=1000000):

    from scipy.stats import poisson
    from collections import Counter

    base = os.path.split(finp)[0]
    print "Base path:", base

    assert wdict is not None, "Word dictionary must be specified for this function to work!"

    words = defaultdict(list)

    store = pd.HDFStore(outstore, "w", complib="blosc", complevel=9)
    store.put("words", pd.DataFrame(), format="table",
              data_columns=["Id"])

    # frequencies of words in overall dict
    wdict["freqs"] = wdict.n * 1. / wdict.n.sum()
    wdict = wdict[wdict.n > 10]

    # 1000 most common words
    wdict_hottest = wdict.sort_values(by="n", ascending=False).iloc[:1000]

    n = 0
    for post in IterateZippedXML(finp):

        n += 1

        # read-in limit reached
        if limit and n > limit:
            break

        entrydict = dict(ParsePostFromXML(post))

        entrydict["Body_unesc"] = UnescapeHTML(entrydict["Body"].decode("utf-8"))

        ws, ratio = GetRelevantWords(entrydict["Body_unesc"], get_ratio=True)
        nws = len(ws)

        ws_dict = Counter(ws)

        multiprob = 1
        hotindices = []

        for w, mult in ws_dict.items():
            ind = np.where(wdict.words.values == w)[0]
            if len(ind) > 0:
                ind = ind[0]

                freq = wdict.iloc[ind].freqs
                multiprob *= poisson.pmf(mult, freq * nws, loc=0)

                ind_hot = np.where(wdict_hottest.values == w)[0]
                if len(ind_hot) > 0:
                    hotindices.append(ind_hot[0])

        words["ratios"].append(ratio)
        words["probs"].append(multiprob)
        words["hot_indices"].append(hotindices)
        words["Id"].append(entrydict["Id"])

        if n % 1000 == 0:

            df = pd.DataFrame(words)
            store.append("words", df, format="table", data_columns=["Id"])
            words.clear()
            del df

            print "Processed %i posts." % n

    # we need to push the remainder of posts left in the dictionary
    if words.values() != []:

        df = pd.DataFrame(words)
        store.append("words", df, format="table", data_columns=["Id"])
        words.clear()
        del df

    # ...and always close the store ;)
    store.close()

    return True


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

    # build word lists
    BuildWordLists(f, "./words_observed_.hdf5",
                   pd.HDFStore("words.hdf5", "r", complib="blosc", complevel=9).get("all"))
