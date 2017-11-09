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

try:
    from spacy.en import STOP_WORDS as STOPWORDS
except:
    from spacy.en import STOPWORDS

import unicodedata

try:
    import spacy
    nlp = spacy.load("en_core_web_md")
except:
    from spacy.en import English
    nlp = English()


def GetRelevantWords(post, get_ratio=False, debug=False):
    """
    Yields relevant words in post after some cleaning.
    Uses spacy to select words.
    """
    p_clean = post.replace("\n", " ")

    # non-greedy code removal
    p_clean = re.sub(r"<code>.*?</code>", " ", p_clean)
    p_clean = re.sub(r"<\/?\w*>", " ", p_clean)
    p_clean = p_clean.replace(":", "").lower()

    if debug:
        print "Cleaned:"
        print p_clean

    words = defaultdict(int)

    n_noun = 0
    n_verb = 0
    n_words = 0

    for word in nlp(p_clean):
        if word.lemma_ not in STOPWORDS and word.text not in STOPWORDS:
            if word.pos_ in ["NOUN", "VERB", "ADV", "ADJ"]:
                if debug:
                    print "%s, %s, %s" % (word, word.lemma_, word.pos_)

                # some custom skipping
                if "noreferrer" in word.lemma_ or "nofollow" in word.lemma_:
                    continue
                if word.lemma_ in ["<", ">"]:
                    continue

                words[unicodedata.normalize("NFKD", word.lemma_).encode("ascii", "ignore")] += 1
                n_words += 1
                if get_ratio:
                    if word.pos_ == "NOUN":
                        n_noun += 1
                    if word.pos_ == "VERB":
                        n_verb += 1

    if get_ratio:
        if n_noun > 0:
            return words, n_words, n_verb * 1. / n_noun
        else:
            return words, n_words, -1.
    else:
        return words, n_words


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

        words, _ = GetRelevantWords(entrydict["Body_unesc"])
        for w, mult in words:
            word_dict[w] += mult

        if n % 5000 == 0:
            print n, len(word_dict.keys())

    # saving to hdf
    words = pd.DataFrame({"words": word_dict.keys(), "n": word_dict.values()})
    store = pd.HDFStore(outp, "w", complib="blosc", complevel=9)
    store.put("all", words)
    store.close()

    return True


# builds dictionary of all occurring words
def BuildDictionariesFromDB(instore_path, indb_path, outstore_path,
                            start=0, stop=100000, onlyquestions=False):

    base = os.path.split(instore_path)[0]
    print "Base path:", base

    word_dict = defaultdict(int)

    # saving to hdf
    # outstore
    outstore = pd.HDFStore(outstore_path, "w", complib="blosc", complevel=9)
    # outstore.put("dict", pd.DataFrame(), format="table", data_columns=True)
    outstore.close()

    # instore
    instore = pd.HDFStore(instore_path, "r", complib="blosc", complevel=9)
    chunks = instore.select("posts", chunksize=10000, start=start, stop=stop,
                            iterator=True)

    # db with all posts
    conn = sqlite3.connect(indb_path)

    n = 0
    for chunk in chunks:

        for i in range(chunk.shape[0]):

            pid = chunk.iloc[i].Id

            if onlyquestions and (chunk.iloc[i].PostTypeId != 1):
                continue

            n += 1

            post = conn.execute("SELECT post FROM posts WHERE id=?", (pid,)).fetchall()[0][0]

            ws, nws = GetRelevantWords(post, get_ratio=False)

            for w, mult in ws.items():
                word_dict[w] += mult

            if n % 10000 == 0:
                new = pd.DataFrame({"words": word_dict.keys(), "n": word_dict.values()})

                outstore.open()
                if "dict" in outstore:
                    old = outstore.get("dict")
                    if old.shape[0] > 0:
                        new = pd.merge(new, old, on="words", suffixes=("", "_r"), how="outer")
                        new["n"] = new.n.add(new.n_r, fill_value=0)
                        del new["n_r"]
                    del old

                outstore.put("dict", new, format="table", data_columns=True)

                print "#Entry: %i, #Unique Words: %i, #Words: %i" % (n, new.shape[0], new.n.sum())

                outstore.close()
                word_dict.clear()
                del new

    # we need to push the remainder of posts left in the dictionary
    if word_dict.values() != []:

        new = pd.DataFrame({"words": word_dict.keys(), "n": word_dict.values()})

        outstore.open()
        if "dict" in outstore:
            old = outstore.get("dict")
            if old.shape[0] > 0:
               new = pd.merge(new, old, on="words", suffixes=("", "_r"), how="outer")
               new["n"] = new.n.add(new.n_r, fill_value=0)
               del new["n_r"]
            del old

        outstore.put("dict", new, format="table", data_columns=True)

        print "#Entry: %i, #Unique Words: %i, #Words: %i" % (n, new.shape[0], new.n.sum())

        outstore.close()
        word_dict.clear()
        del new

    instore.close()

    return True


def MergeDictionaries(pathlist, outpath):

    outstore = pd.HDFStore(outpath, "w", complib="blosc", complevel=9)

    outdf = None

    for i, pathi in enumerate(pathlist):

        print "Processing dataframe %i of %i (%s)." % (i, len(pathlist), pathi)

        instorei = pd.HDFStore(pathi, "r", complib="blosc", complevel=9)

        dfi = instorei.get("dict")
        dfi.set_index("words", inplace=True)
        dfi = dfi[~dfi.index.duplicated()]

        if outdf is None:
            outdf = dfi
        else:
            dfi.rename(columns={"n": "n_new"}, inplace=True)
            # outdf["n"] = outdf.n.add(dfi.n, axis="index", fill_value=0)
            outdf = outdf.combine_first(dfi)
            outdf[np.isnan(outdf)] = 0
            outdf["n"] = outdf.n + outdf.n_new
            del outdf["n_new"]

        instorei.close()

    print "Writing merged output df to %s." % outpath
    outstore.put("dict", outdf.reset_index())
    outstore.close()


def BuildWordLists(instore_path, wdict_path, indb_path, outstore_path,
                   limit=1000000, order_cut=20000000,
                   onlyquestions=True):

    from scipy.stats import poisson, multinomial

    words = defaultdict(list)

    # outstore
    outstore = pd.HDFStore(outstore_path, "w", complib="blosc", complevel=9)
    # outstore.put("words", pd.DataFrame(), format="table",
    #              data_columns=["Id", "nwords", "ratio"],
    #              min_itemsize={"hot_indices": 800})

    wdict = pd.HDFStore(wdict_path, "r", complib="blosc", complevel=9).get("dict")

    # frequencies of words in overall dict
    wdict["freqs"] = wdict.n * 1. / wdict.n.sum()

    wdict = wdict.sort_values(by="n", ascending=False)
    wdict["order"] = np.arange(1, wdict.shape[0] + 1)

    # take only words that describe 95 % of sum of all words
    cutoff = np.where((wdict.n.cumsum() * 1. / wdict.n.sum()) > 0.95)[0][0]
    wdict = wdict.iloc[:cutoff]
    wdict.set_index("words", inplace=True, drop=False)

    # instore
    instore = pd.HDFStore(instore_path, "r", complib="blosc", complevel=9)
    chunks = instore.select("posts", chunksize=10000, iterator=True)

    # db with all posts
    conn = sqlite3.connect(indb_path)

    n = 0
    for chunk in chunks:

        if n > limit:
            break

        for i in range(chunk.shape[0]):

            pid = chunk.iloc[i].Id

            if onlyquestions and (chunk.iloc[i].PostTypeId != 1):
                continue

            n += 1

            # read-in limit reached
            if n > limit:
                break

            post = conn.execute("SELECT post FROM posts WHERE id=?", (pid,)).fetchall()[0][0]

            ws, nws, ratio = GetRelevantWords(post, get_ratio=True)

            wsdf = pd.DataFrame({"w": ws.keys(), "mult": ws.values()})
            wsdf.set_index("w", inplace=True, drop=False)
            wsdf = wsdf.join(wdict, how="inner")

            prob_poisson = np.prod(poisson.pmf(wsdf.mult, nws * wsdf.freqs))
            prob_bern = np.prod((nws * wsdf.freqs) ** np.minimum(1, wsdf.mult) * (1 - nws * wsdf.freqs) ** (1 - np.minimum(1, wsdf.mult)))
            # multiprob = multinomial.pmf(wsdf.mult, nws, p=nws * wsdf.freqs)
            hotindices = wsdf[wsdf.order < order_cut].order.values

            words["Id"].append(pid)
            words["ratio"].append(float(ratio))
            words["nwords"].append(int(nws))
            # words["prob_multi"].append(multiprob)
            words["prob_bern"].append(float(prob_bern))
            words["prob_poiss"].append(float(prob_poisson))
            words["ordersum"].append(float(wsdf.order.sum()))
            words["ordermean"].append(float(wsdf.order.mean()))
            words["orderstd"].append(float(wsdf.order.std()))
            # words["hot_indices"].append(";".join(map(str, sorted(hotindices)))[:500])
            words["hot_indices"].append(";".join(map(str, hotindices))[:800])

            if n % 1000 == 0:

                df = pd.DataFrame(words)
                # if "words" in outstore:
                #     print outstore.select("words").dtypes
                #     print df.dtypes
                outstore.append("words", df, format="table", data_columns=["Id", "nwords", "ratio"],
                                min_itemsize={"hot_indices": 800})
                words.clear()
                del df

                print "Processed %i posts of %s." % (n, instore_path)

    # we need to push the remainder of posts left in the dictionary
    if words.values() != []:

        df = pd.DataFrame(words)
        outstore.append("words", df, format="table", data_columns=["Id", "nwords", "ratio"],
                        min_itemsize={"hot_indices": 800})
        words.clear()
        del df

    # ...and always close the stores ;)
    outstore.close()
    instore.close()

    return True


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("Stackoverflow word analysis steered by config file.")
    parser.add_argument("fn", nargs=1, metavar="<config file>",
                        help="Configuration file containing desired paths and settings.")
    args = parser.parse_args()

    cfgfn = args.fn[0]

    # importing things from file
    cfg = local_import(cfgfn)

    # making sure that relevant directories exist
    for k, p in cfg.paths.items():
        if not os.path.exists(p):
            print "Creating non-existing directory {0}.".format(p)
            os.makedirs(p)

    def BuildDicts(year):
        chunks = np.arange(0, 1100000, 200000)
        for ic, cstart in enumerate(chunks[:-1]):
            cend = chunks[ic + 1]
            BuildDictionariesFromDB(os.path.join(cfg.paths["caches"], "posts_%s.hdf5" % year), cfg.paths["db"],
                                    os.path.join(cfg.paths["dictionaries"], "dict_%s_%s_%s.hdf5" % (year, cstart + 1, cend)),
                                    start=cstart + 1, stop=cend)

    def BuildLists(year):
        BuildWordLists(os.path.join(cfg.paths["caches"], "posts_%s.hdf5" % year),
                       os.path.join(cfg.paths["dictionaries"], "merged.hdf5"),
                       cfg.paths["db"],
                       os.path.join(cfg.paths["features"], "features_%s.hdf5" % year))

    # map(BuildDicts, [2008])

    allyears = range(2008, 2018)
    # pmap(BuildDicts, allyears, numprocesses=2)
    pmap(BuildLists, [2016, 2017], numprocesses=2)

    # merge individual (yearly) dicts together
    # MergeDictionaries(glob("/home/alex/data/stack_cache/dictionaries_new/*.hdf5"),
    #                   os.path.join(cfg.paths["dictionaries"], "merged.hdf5"))
