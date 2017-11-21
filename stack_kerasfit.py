#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Fit models to posts / titles / etc using embeddings and keras
#
#
from stack_nlp import *
from gensim.models import KeyedVectors
from keras.utils import to_categorical


def GetAnswerTimeQuantiles(df, ncat):
    # the bins assuming that one bin for not-answered questions is added
    timecat_bins = np.linspace(-0.5, ncat + 0.5, ncat + 2)

    tmask = np.isfinite(df.dt_accanswer_hour)
    time_categories = mquantiles(df.loc[tmask].dt_accanswer_hour, prob=np.linspace(0, 1, ncat + 1))
    return time_categories, timecat_bins


def AddTimeCategories(df, timequants):
    tmask = np.isfinite(df.dt_accanswer_hour)
    df["timecat"] = 0
    df.loc[tmask, "timecat"] = np.digitize(df.loc[tmask].dt_accanswer_hour, timequants) - 1
    df.loc[~tmask, "timecat"] = len(timequants) - 1


def GetMostCommonTags(df, n=20):
    """ Get the most common tags of questions in a df."""
    from collections import defaultdict
    c = defaultdict(int)

    def add(t):
        c[t] += 1
    df.Tags.apply(lambda x: [add(t) for t in x])

    skeys = sorted(c, key=c.get, reverse=True)
    tagdf = pd.DataFrame({"tags": skeys, "counts": [c[sk] for sk in skeys]})
    tagdf.set_index("tags", inplace=True)
    return tagdf.iloc[:n]


def FittingFriend(cfg):

    print "Importing and preparing data..."
    PrepareData(cfg)
    data = cfg.data
    qs = data["meta"]
    conn = data["dbconn"]

    for fit in cfg.fits:
        print "\n>> Working on fit of type %s with name %s" % (fit["type"], fit["name"])

        assert "embed_path" in fit, "Embedding input is not defined! This is currently required!"

        if not os.path.exists(fit["embed_out"]):
            ConvertToGensimFile(fit["embed_path"], fit["embed_out"])

        assert "labelfct" in fit, "Necessary to provide label function!"

        if fit.get("create_common_tags", False):  # creating most common tags
            mostcommon = GetMostCommonTags(qs, n=1000)
            mostcommon.to_csv("./infos/most_common_tags.csv")
            a = pd.HDFStore("./infos/most_common_tags.hdf5", mode="w", complib="blosc", complevel=9)
            a.put("tags", mostcommon)
            a.close()
            return 0.

        print "Calculating labels according to provided label function..."
        qs["label"] = fit["labelfct"](qs)
        nsample = fit.get("nsample", 100000)

        if fit.get("uniform", True):

            print "Selecting a sample of %i posts uniformly and randomly within each group." % nsample
            qssel = SelectUniformlyFromColumn(qs, "label", n=nsample)

        else:
            print "Selecting a sample of %i posts randomly." % nsample
            qssel = qs.sample(nsample)

        if fit.get("binary", False):
            nouts = 1
        else:
            nouts = to_categorical(qssel["label"]).shape[1]

        qstrain = qssel.iloc[:int(0.8 * qssel.shape[0])]
        qstest = qssel.iloc[int(0.8 * qssel.shape[0]):]
        print "Length of the training set:", len(qstrain)
        print "Length of the testing set:", len(qstest)

        print "Output label dimensions:", nouts

        print "Opening embedding vectors"
        gmod = KeyedVectors.load_word2vec_format(fit["embed_out"], binary=False)

        inp_data = []
        inp_test_data = []

        word_tokenizer = False
        if fit.get("posts", True):

            print "Retrieving relevant posts for training and testing."
            posts_train = GetDBPosts(qstrain.Id.values, conn)
            posts_test = GetDBPosts(qstest.Id.values, conn)

            if fit.get("clean", False):
                print "Cleaning posts..."

                try:
                    from spacy.lang.en import STOP_WORDS as STOPWORDS
                except:
                    from spacy.en import STOPWORDS

                posts_train = [[w for w in p if p not in STOPWORDS] for p in posts_train]
                posts_test = [[w for w in p if p not in STOPWORDS] for p in posts_test]
            else:
                print "Warning! Posts are not cleaned! (stop-words, lemmatization etc)"

            print "Fitting tokenizer..."
            word_tokenizer = Tokenizer(fit["nfeatures"])
            word_tokenizer.fit_on_texts(posts_train)

            print "Tokenizing..."
            posts_train_tf = word_tokenizer.texts_to_sequences(posts_train)
            posts_test_tf = word_tokenizer.texts_to_sequences(posts_test)

            embed()

            maxlen_posts = 600  # this catches roughly 95 % of all posts
            print "Padding to length %i..." % maxlen_posts
            posts_train_tf = pad_sequences(posts_train_tf, maxlen=maxlen_posts,
                                           padding="post", truncating="post")
            posts_test_tf = pad_sequences(posts_test_tf, maxlen=maxlen_posts,
                                          padding="post", truncating="post")

            inp_data.append(posts_train_tf)
            inp_test_data.append(posts_test_tf)

        if fit.get("titles", True):
            print "Retrieving relevant titles for training and testing."
            titles_train = np.squeeze(qstrain.Title.values)
            titles_test = np.squeeze(qstest.Title.values)

            if not word_tokenizer:
                print "Building tokenizer on titles."
                word_tokenizer = Tokenizer(fit["nfeatures"])
                word_tokenizer.fit_on_texts(titles_train)

            titles_train_tf = word_tokenizer.texts_to_sequences(titles_train)
            titles_test_tf = word_tokenizer.texts_to_sequences(titles_test)

            maxlen_titles = 30  # catches all sentences in training set checked on 17/11
            titles_train_tf = pad_sequences(titles_train_tf, maxlen=maxlen_titles, padding="post", truncating="post")
            titles_test_tf = pad_sequences(titles_test_tf, maxlen=maxlen_titles, padding="post", truncating="post")

            inp_data.append(titles_train_tf)
            inp_test_data.append(titles_test_tf)

        # setting up weights matrix for embedding in keras
        weights_matrix = np.zeros((fit["nfeatures"] + 1, fit["embed_dim"]))
        for word, i in word_tokenizer.word_index.items():

            if i > fit["nfeatures"]:
                continue
            try:
                embedding_vector = gmod.word_vec(word)
                if embedding_vector is not None:
                    weights_matrix[i] = embedding_vector
            except:
                weights_matrix[i] = np.zeros(fit["embed_dim"])

        pools = []
        outs = []
        inps = []

        if fit.get("posts", True):

            posts_input = Input(shape=(maxlen_posts,), name="posts_input")
            posts_embedding = Embedding(fit["nfeatures"] + 1, fit["embed_dim"],
                                        weights=[weights_matrix])(posts_input)
            pools.append(GlobalAveragePooling1D()(posts_embedding))
            outs.append(Dense(nouts, activation="sigmoid" if nouts == 1 else "softmax",
                              name="posts_reg_out")(pools[-1]))
            inps.append(posts_input)

        if fit.get("titles", True):

            titles_input = Input(shape=(maxlen_titles,), name="titles_input")
            titles_embedding = Embedding(fit["nfeatures"] + 1, fit["embed_dim"],
                                         weights=[weights_matrix])(titles_input)
            pools.append(GlobalAveragePooling1D()(titles_embedding))
            outs.append(Dense(nouts, activation="sigmoid" if nouts == 1 else "softmax",
                              name="titles_reg_out")(pools[-1]))
            inps.append(titles_input)

        meta_embedding_dims = 64
        for feat in fit.get("cat_features", []):

            feat_input = Input(shape=(1,), name="%s_input_cat" % feat)
            feat_embedding = Embedding(max(qssel[feat]) + 1, meta_embedding_dims)(feat_input)
            pools.append(Reshape((meta_embedding_dims,))(feat_embedding))
            inps.append(feat_input)

            inp_data.append(qstrain[feat])
            inp_test_data.append(qstest[feat])

        for feat in fit.get("features", []):
            feat_input = Input(shape=(1,), name="%s_input" % feat)
            pools.append(feat_input)
            inps.append(feat_input)

            inp_data.append(qstrain[feat])
            inp_test_data.append(qstest[feat])

        merged = concatenate(pools)

        if fit.get("cnn", False):
            print "Using CNN layer in network, please check options for filter and kernel size."
            merged = Conv1D(250, 3, padding="valid",
                            activation="relu", strides=1)(merged)
            merged = GlobalMaxPooling1D()(merged)

        hidden_1 = Dense(256, activation="relu")(merged)
        hidden_1 = BatchNormalization()(hidden_1)

        main_output = Dense(nouts, activation="sigmoid" if nouts == 1 else "softmax",
                            name="main_out")(hidden_1)

        model = Model(inputs=inps, outputs=[main_output] + outs)

        model.compile(loss="binary_crossentropy" if nouts == 1 else "categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"],
                      loss_weights=[1, 0.2, 0.2])

        print model.summary()

        plot_model(model, to_file='./plots/fit_%s.pdf' % fit["id"])
        plot_model(model, to_file='./plots/fit_%s_shapes.pdf' % fit["id"], show_shapes=True)

        if nouts == 1:
            print "No-information baselines for each group:"
            print "Training:", 1 - np.sum(qstrain["label"]) * 1. / qstrain["label"].shape[0]
            print "Testing:", 1 - np.sum(qstest["label"]) * 1. / qstest["label"].shape[0]
            print "Validation:", 1 - np.mean(qstrain["label"][:(int(posts_train_tf.shape[0] * fit["nsplit"]))])

        csv_logger = CSVLogger("./logging/training_%s.csv" % fit["id"])

        # from keras.callbacks import EarlyStopping
        # early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

        convert_dims = lambda x: to_categorical(x, num_classes=nouts) if nouts > 1 else x

        try:
            model.fit(inp_data, [convert_dims(qstrain["label"]) for _ in xrange(len(outs) + 1)],
                      batch_size=fit["nbatch"], epochs=fit["nepoch"],
                      validation_split=fit["nsplit"], callbacks=[csv_logger])
        except KeyboardInterrupt:
            print "Stopping fit process, current result should be kept!"

        a = model.evaluate(x=inp_test_data,
                           y=[convert_dims(qstest["label"]) for _ in xrange(len(outs) + 1)])
        print "Testing results:", a

        if fit.get("save", False):

            model.save("./models/keras_full_%s.keras" % fit["id"])
            model.save_weights("./models/keras_weights_%s.keras" % fit["id"])

        embed()


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

    FittingFriend(cfg)
