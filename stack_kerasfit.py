#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Fit models to posts / titles / etc using embeddings and keras
#
#
from stack_nlp import *
from gensim.models import KeyedVectors


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


def FittingFriend(cfg):

    print "Importing and preparing data..."
    PrepareData(cfg)
    data = cfg.data
    qs = data["meta"]
    conn = data["dbconn"]

    for fit in cfg.fits:
        print "Working on fit of type %s with name %s" % (fit["type"], fit["name"])

        assert "embed_path" in fit, "Embedding input is not defined! This is currently required!"

        if not os.path.exists(fit["embed_out"]):
            ConvertToGensimFile(fit["embed_path"], fit["embed_out"])

        print "Opening embedding vectors"
        gmod = KeyedVectors.load_word2vec_format(fit["embed_out"], binary=False)

        assert "labelfct" in fit, "Necessary to provide label function!"

        # calculating labels
        qs["label"] = fit["labelfct"](qs)
        nsample = fit.get("nsample", 100000)

        if fit.get("uniform", True):

            print "Selecting a sample of %i posts uniformly and randomly within each group." % nsample
            qssel = SelectUniformlyFromColumn(qs, "label", n=nsample)

        else:
            print "Selecting a sample of %i posts randomly." % nsample
            qssel = qs.sample(nsample)

        qstrain = qssel.iloc[:int(0.8 * qssel.shape[0])]
        qstest = qssel.iloc[int(0.8 * qssel.shape[0]):]
        print "Length of the training set:", len(qstrain)
        print "Length of the testing set:", len(qstest)

        inp_data = []

        word_tokenizer = False
        if fit.get("posts", True):

            print "Retrieving relevant posts for training and testing."
            posts_train = GetDBPosts(qstrain.Id.values, conn)
            posts_test = GetDBPosts(qstest.Id.values, conn)
            conn.close()

            print "Fitting tokenizer..."
            word_tokenizer = Tokenizer(fit["nfeatures"])
            word_tokenizer.fit_on_texts(posts_train)

            print "Tokenizing..."
            posts_train_tf = word_tokenizer.texts_to_sequences(posts_train)
            posts_test_tf = word_tokenizer.texts_to_sequences(posts_test)

            maxlen_posts = 600  # this catches roughly 95 % of all posts
            print "Padding to length %i..." % maxlen_posts
            posts_train_tf = pad_sequences(posts_train_tf, maxlen=maxlen_posts,
                                           padding="post", truncating="post")
            posts_test_tf = pad_sequences(posts_test_tf, maxlen=maxlen_posts,
                                          padding="post", truncating="post")

            inp_data.append(posts_train_tf)

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
            outs.append(Dense(1, activation="sigmoid", name="posts_reg_out")(pools[-1]))
            inps.append(posts_input)

        if fit.get("titles", True):

            titles_input = Input(shape=(maxlen_titles,), name="titles_input")
            titles_embedding = Embedding(fit["nfeatures"] + 1, fit["embed_dim"],
                                         weights=[weights_matrix])(titles_input)
            pools.append(GlobalAveragePooling1D()(titles_embedding))
            outs.append(Dense(1, activation="sigmoid", name="titles_reg_out")(pools[-1]))
            inps.append(titles_input)

        meta_embedding_dims = 64
        for feat in fit.get("cat_features", []):

            feat_input = Input(shape=(1,), name="%s_input_cat" % feat)
            feat_embedding = Embedding(max(qssel[feat]) + 1, meta_embedding_dims)(feat_input)
            pools.append(Reshape((meta_embedding_dims,))(feat_embedding))
            inps.append(feat_input)

            inp_data.append(qstrain[feat])

        for feat in fit.get("features", []):
            feat_input = Input(shape=(1,), name="%s_input" % feat)
            pools.append(feat_input)
            inps.append(feat_input)

        merged = concatenate(pools)

        hidden_1 = Dense(256, activation="relu")(merged)
        hidden_1 = BatchNormalization()(hidden_1)

        main_output = Dense(1, activation="sigmoid", name="main_out")(hidden_1)

        model = Model(inputs=inps, outputs=[main_output] + outs)

        model.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"],
                      loss_weights=[1, 0.2, 0.2])

        print model.summary()

        plot_model(model, to_file='./plots/fit_%s.pdf' % fit["id"])
        plot_model(model, to_file='./plots/fit_%s_shapes.pdf' % fit["id"], show_shapes=True)

        print "No-information baselines for each group:"
        print "Training:", 1 - np.sum(qstrain["label"]) * 1. / qstrain["label"].shape[0]
        print "Testing:", 1 - np.sum(qstest["label"]) * 1. / qstest["label"].shape[0]
        print "Validation:", 1 - np.mean(qstrain["label"][:(int(posts_train_tf.shape[0] * fit["nsplit"]))])

        csv_logger = CSVLogger("./logging/training_%s.csv" % fit["id"])

        # from keras.callbacks import EarlyStopping
        # early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

        model.fit(inp_data, [qstrain["label"] for _ in xrange(len(outs) + 1)],
                  batch_size=fit["nbatch"], epochs=fit["nepoch"],
                  validation_split=fit["nsplit"], callbacks=[csv_logger])

        # a = model.evaluate(x=[posts_test_tf, titles_test_tf, qstest.dayhour.values, qstest.weekday.values, qstest.day.values],
        #                    y=[qstest[label] for _ in xrange(len(outs) + 1)])

        print "Testing results:", a

        if fit.get("save", False):

            model.save("./models/keras_full_%s.keras" % cfg["id"])
            model.save_weights("./models/keras_weights_%s.keras" % cfg["id"])

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
