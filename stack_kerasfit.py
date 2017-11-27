#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  Fit models to posts / titles / etc using embeddings and keras
#
#
from stack_nlp import *
from gensim.models import KeyedVectors
from keras.utils import to_categorical
import dill


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


def DataImportNeeded(cfg):
    for fitcfg in cfg.fits:
        if not fitcfg.get("from_cache", False):
            return True
    return False


def FittingFriend(cfg):

    if DataImportNeeded(cfg):
        print "Importing and preparing data..."
        PrepareData(cfg)
        data = cfg.data
        qs = data["meta"]
        conn = data["dbconn"]

    for fitcfg in cfg.fits:
        print "\n>> Working on fit of type %s with name %s" % (fitcfg["type"], fitcfg["name"])

        if not fitcfg.get("from_cache", False):

            assert "embed_path" in fitcfg, "Embedding input is not defined! This is currently required!"

            if not os.path.exists(fitcfg["embed_out"]):
                print "Embedding word2vec file does not exist yet, attempting conversion!"
                ConvertToGensimFile(fitcfg["embed_path"], fitcfg["embed_out"])

            assert "labelfct" in fitcfg, "Necessary to provide label function!"

            if fitcfg.get("create_common_tags", False):  # creating most common tags
                mostcommon = GetMostCommonTags(qs, n=1000)
                mostcommon.to_csv("./infos/most_common_tags.csv")
                a = pd.HDFStore("./infos/most_common_tags.hdf5", mode="w", complib="blosc", complevel=9)
                a.put("tags", mostcommon)
                a.close()
                return 0.

            print "Calculating labels according to provided label function..."
            qs["label"] = fitcfg["labelfct"](qs, fitcfg)
            nsample = fitcfg.get("nsample", 100000)
            seed = fitcfg.get("seed", np.random.randint(1e6))

            if fitcfg.get("uniform", True):

                print "Selecting a sample of %i posts uniformly and randomly within each group." % nsample
                qssel = SelectUniformlyFromColumn(qs, "label", n=nsample, seed=seed)

            else:
                print "Selecting a sample of %i posts randomly." % nsample
                qssel = qs.sample(nsample, random_state=seed)

            if fitcfg.get("binary", False):
                nouts = 1
            else:
                nouts = to_categorical(qssel["label"]).shape[1]

            qstrain = qssel.iloc[:int(0.8 * qssel.shape[0])]
            qstest = qssel.iloc[int(0.8 * qssel.shape[0]):]
            print "Length of the training set:", len(qstrain)
            print "Length of the testing set:", len(qstest)

            print "Output label dimensions:", nouts

            print "Opening embedding vectors"
            gmod = KeyedVectors.load_word2vec_format(fitcfg["embed_out"], binary=not fitcfg.get("train_embeddings", True))

            inp_data = []
            inp_test_data = []

            word_tokenizer = False
            if fitcfg.get("posts", True):

                print "Retrieving relevant posts for training and testing."
                if not fitcfg.get("use_saved_posts", False):
                    posts_train = GetDBPosts(qstrain.Id.values, conn)
                    posts_test = GetDBPosts(qstest.Id.values, conn)
                    print "Pickling posts for training and testing to files..."
                    dill.dump(posts_train, open("./models/posts_train_%s.dill" % fitcfg["id"], "w"))
                    dill.dump(posts_test, open("./models/posts_test_%s.dill" % fitcfg["id"], "w"))
                else:
                    print "Using pickled posts."
                    posts_train = dill.load(open("./models/posts_train_%s.dill" % fitcfg["id"], "r"))
                    posts_test = dill.load(open("./models/posts_test_%s.dill" % fitcfg["id"], "r"))

                if fitcfg.get("clean", False):
                    print "Cleaning posts..."
                    posts_train = [TextCleansing(p) for p in posts_train]
                    posts_test = [TextCleansing(p) for p in posts_test]
                else:
                    print "Warning! Posts are not cleaned! (stop-words, lemmatization etc)"

                if not fitcfg.get("tokenizer", False):
                    print "Fitting tokenizer..."
                    word_tokenizer = Tokenizer(fitcfg["nfeatures"])
                    word_tokenizer.fit_on_texts(posts_train + posts_test)
                    print "Dumping tokenizer to file for later use."
                    dill.dump(word_tokenizer, open("./models/tokenizer_%s.dill" % fitcfg["id"], "w"))
                else:
                    print "Using pre-fitted tokenizer %s..." % fitcfg["tokenizer"]
                    word_tokenizer = dill.load(open(fitcfg["tokenizer"], "r"))

                print "Tokenizing..."
                posts_train_tf = word_tokenizer.texts_to_sequences(posts_train)
                posts_test_tf = word_tokenizer.texts_to_sequences(posts_test)

                # maxlen_posts = 600  # this catches roughly 95 % of all posts without text cleaning
                maxlen_posts = 500  # this catches roughly 95 % of all posts with text cleaning
                print "Padding to length %i..." % maxlen_posts
                posts_train_tf = pad_sequences(posts_train_tf, maxlen=maxlen_posts,
                                               padding="post", truncating="post")
                posts_test_tf = pad_sequences(posts_test_tf, maxlen=maxlen_posts,
                                              padding="post", truncating="post")

                inp_data.append(posts_train_tf)
                inp_test_data.append(posts_test_tf)

            if fitcfg.get("titles", True):
                print "Retrieving relevant titles for training and testing."
                titles_train = np.squeeze(qstrain.Title.values)
                titles_test = np.squeeze(qstest.Title.values)

                if fitcfg.get("clean", False):
                    print "Cleaning titles..."
                    titles_train = [TextCleansing(t) for t in titles_train]
                    titles_test = [TextCleansing(t) for t in titles_test]
                else:
                    print "Warning! Titles are not cleaned! (stop-words, lemmatization etc)"

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
            weights_matrix = np.zeros((fitcfg["nfeatures"] + 1, fitcfg["embed_dim"]))
            for word, i in word_tokenizer.word_index.items():

                if i > fitcfg["nfeatures"]:
                    continue
                try:
                    embedding_vector = gmod.word_vec(word)
                    if embedding_vector is not None:
                        weights_matrix[i] = embedding_vector
                except:
                    weights_matrix[i] = np.zeros(fitcfg["embed_dim"])

            pools = []
            outs = []
            inps = []

            if fitcfg.get("posts", True):

                posts_input = Input(shape=(maxlen_posts,), name="posts_input")
                posts_embedding = Embedding(fitcfg["nfeatures"] + 1, fitcfg["embed_dim"],
                                            weights=[weights_matrix],
                                            input_length=maxlen_posts,
                                            trainable=fitcfg.get("train_embeddings", True))(posts_input)

                if fitcfg.get("cnn", False):
                    print "Using CNN layer in network, please check options for filter and kernel size."
                    posts_embedding = Conv1D(250, 3, padding="valid",
                                             activation="relu", strides=1)(posts_embedding)

                pools.append(GlobalAveragePooling1D()(posts_embedding))
                outs.append(Dense(nouts, activation="sigmoid" if nouts == 1 else "softmax",
                                  name="posts_reg_out")(pools[-1]))
                inps.append(posts_input)

            if fitcfg.get("titles", True):

                titles_input = Input(shape=(maxlen_titles,), name="titles_input")
                titles_embedding = Embedding(fitcfg["nfeatures"] + 1, fitcfg["embed_dim"],
                                             weights=[weights_matrix],
                                             input_length=maxlen_titles,
                                             trainable=fitcfg.get("train_embeddings", True))(titles_input)

                if fitcfg.get("cnn", False):
                    print "Using CNN layer in network, please check options for filter and kernel size."
                    titles_embedding = Conv1D(250, 3, padding="valid",
                                              activation="relu", strides=1)(titles_embedding)

                pools.append(GlobalAveragePooling1D()(titles_embedding))
                outs.append(Dense(nouts, activation="sigmoid" if nouts == 1 else "softmax",
                                  name="titles_reg_out")(pools[-1]))
                inps.append(titles_input)

            meta_embedding_dims = 64
            for feat in fitcfg.get("cat_features", []):

                feat_input = Input(shape=(1,), name="%s_input_cat" % feat)
                feat_embedding = Embedding(max(qssel[feat]) + 1, meta_embedding_dims)(feat_input)
                pools.append(Reshape((meta_embedding_dims,))(feat_embedding))
                inps.append(feat_input)

                inp_data.append(qstrain[feat])
                inp_test_data.append(qstest[feat])

            for feat in fitcfg.get("features", []):
                feat_input = Input(shape=(1,), name="%s_input" % feat)
                pools.append(feat_input)
                inps.append(feat_input)

                inp_data.append(qstrain[feat])
                inp_test_data.append(qstest[feat])

            merged = concatenate(pools)

            # if fitcfg.get("LSTM", False):
                # model.add(LSTM(int(document_max_num_words*1.5), input_shape=(document_max_num_words, num_features)))
                # model.add(Dropout(0.3))

            hidden_1 = Dense(256, activation="relu")(merged)

            if fitcfg.get("dropout", False):
                hidden_1 = Dropout(0.2)(hidden_1)

            hidden_1 = BatchNormalization()(hidden_1)

            main_output = Dense(nouts, activation="sigmoid" if nouts == 1 else "softmax",
                                name="main_out")(hidden_1)

            model = Model(inputs=inps, outputs=[main_output] + outs)

            model.compile(loss="binary_crossentropy" if nouts == 1 else "categorical_crossentropy",
                          optimizer="adam",
                          metrics=["accuracy"],
                          loss_weights=[1, 0.2, 0.2])

            print model.summary()

            plot_model(model, to_file='./plots/fit_%s.pdf' % fitcfg["id"])
            plot_model(model, to_file='./plots/fit_%s_shapes.pdf' % fitcfg["id"], show_shapes=True)

            if nouts == 1:
                print "No-information baselines for each group:"
                print "Training:", 1 - np.sum(qstrain["label"]) * 1. / qstrain["label"].shape[0]
                print "Testing:", 1 - np.sum(qstest["label"]) * 1. / qstest["label"].shape[0]
                print "Validation:", 1 - np.mean(qstrain["label"][:(int(posts_train_tf.shape[0] * fitcfg["nsplit"]))])

            csv_logger = CSVLogger("./logging/training_%s.csv" % fitcfg["id"])

            # from keras.callbacks import EarlyStopping
            # early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

            convert_dims = lambda x: to_categorical(x, num_classes=nouts) if nouts > 1 else x
            try:
                model.fit(inp_data, [convert_dims(qstrain["label"]) for _ in xrange(len(outs) + 1)],
                          batch_size=fitcfg["nbatch"], epochs=fitcfg["nepoch"],
                          validation_split=fitcfg["nsplit"], callbacks=[csv_logger])
            except KeyboardInterrupt:
                print "Stopping fit process, current result should be kept!"

            a = model.evaluate(x=inp_test_data,
                               y=[convert_dims(qstest["label"]) for _ in xrange(len(outs) + 1)])
            print "Testing results:", a

            test_truths = qstest["label"]
            test_preds = model.predict(inp_test_data)
            test_df = qstest

            if fitcfg.get("save", False):

                model.save("./models/keras_full_%s.keras" % fitcfg["id"])
                model.save_weights("./models/keras_weights_%s.keras" % fitcfg["id"])

                dill.dump(test_preds, open("./models/test_preds_%s.dill" % fitcfg["id"], "w"))
                dill.dump(test_truths, open("./models/test_truths_%s.dill" % fitcfg["id"], "w"))
                dill.dump(qstest, open("./models/test_df_%s.dill" % fitcfg["id"], "w"))

        else:
            print "Using cached results!"
            from keras.models import load_model

            model = load_model("./models/keras_full_%s.keras" % fitcfg["id"])
            test_truths = dill.load(open("./models/test_truths_%s.dill" % fitcfg["id"], "r"))
            test_preds = dill.load(open("./models/test_preds_%s.dill" % fitcfg["id"], "r"))
            test_df = dill.load(open("./models/test_df_%s.dill" % fitcfg["id"], "r"))

        if fitcfg.get("plots", True):

            try:
                train_log = pd.read_csv("./logging/training_%s.csv" % fitcfg["id"])
            except:
                train_log = None

            print "Making a few plots..."
            if train_log is not None:
                PlotTrainingResult(train_log, fitcfg)

            PlotConfusionMatrix(test_truths, test_preds, fitcfg, labels=fitcfg.get("grouplabels", None))

            if fitcfg.get("type", False) == "keras_embedding_scores":
                PlotPredictionHistograms(test_truths, test_preds, fitcfg)
                PlotPredictionVsLabels(test_df, test_preds, fitcfg)

            if fitcfg.get("type", False) == "keras_embedding_tags":
                fig = plt.figure()
                sfig = fig.add_axes([0.15, 0.11, 0.845, 0.78])
                cfg.mostcommon_tags.set_index("tags").head(fitcfg.get("ntags", 20)).plot.bar(ax=plt.gca())
                plt.ylabel("Number of tagged questions")
                plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d"))
                plt.savefig("./plots/hist_mostcommontags.pdf")


def PlotPredictionVsLabels(df, preds, cfg):

    nclasses = len(np.unique(df.label))

    # result from "best" class
    goodpreds = preds[0].T[nclasses - 1]

    fig = plt.figure()
    sfig = fig.add_axes([0.15, 0.11, 0.845, 0.78])

    plt.xlabel(r"Question score")
    plt.ylabel(r"Estimated probability of high score")

    print "Score bins are set by hand here"
    xbins = [df.Score.min(), 0, 1, 2, 3, 4, 6, 8, 10, 20, df.Score.max()]

    dummybins = np.arange(len(xbins) - 1)
    binnedpreds = []
    for i in range(len(xbins) - 1):
        maski = (df.Score >= xbins[i]) & (df.Score < xbins[i + 1])
        binnedpreds.append(goodpreds[maski])

    ViolinPlot(dummybins, binnedpreds, bins=None,
               axes=sfig, color="k", draw="amv")

    def label_format(dbins):
        labels = []
        for dbin in dbins:
            labels.append(r"[%i, %i)" % (xbins[dbin], xbins[dbin + 1]))
        # print labels
        return labels

    sfig.set_xticks(dummybins)
    sfig.set_xticklabels(label_format(dummybins), rotation=40, ha="right")

    plt.ylim(0., 1.)
    # plt.semilogx()
    plt.savefig("./plots/pred_probs_vs_class_score_%s.pdf" % cfg["id"])


def PlotPredictionHistograms(truths, preds, cfg):

    mpreds = preds[0]
    truevals = np.unique(truths)
    print truevals

    for predi in xrange(len(truevals)):

        plt.figure()
        plt.xlabel(r"Predicted probability to have a %s score" % cfg["grouplabels"][truevals[predi]])
        plt.ylabel(r"Number of predictions (normalized)")

        for trueval in truevals:

            mask = truths == trueval
            goodpreds = mpreds.T[predi]

            plt.hist(goodpreds[mask], ls="-", color=g_carr[trueval + 1], lw=2,
                     range=[0, 1], bins=100,
                     histtype="step", density=True,
                     label=cfg["grouplabels"][trueval])

        plt.xlim(0., 1.)
        plt.legend(loc="best")
        plt.savefig("./plots/pred_probs_vs_groups_%s_%s.pdf" % (cfg["id"], cfg["grouplabels"][truevals[predi]]))


def PlotConfusionMatrix(truths, preds, cfg, labels=None):

    import matplotlib.cm as cm

    if isinstance(labels, list):
        labels = [r"%s" % l for l in labels]

    # one way of calculating confusion matrix
    preds_bin = np.argmax(preds[0], axis=1)
    comp = pd.DataFrame({"truth": truths, "prediction": preds_bin})
    comp = comp.groupby(["truth", "prediction"]).apply(len)
    comp = comp.unstack(level=-1)

    comp[np.isnan(comp)] = 0

    # normalization
    comp = comp.div(comp.sum(axis=1), axis=0)
    comp = comp.T
    comp.sort_index(ascending=False, inplace=True)

    plt.figure(figsize=(15, 12))
    plt.title(r"$P(\mathrm{prediction}\vert\mathrm{truth})$")

    if labels is not None:
        ax = sns.heatmap(comp, annot=False, linewidths=0.5, cmap="binary",
                         xticklabels=labels, yticklabels=labels[::-1], square=True,
                         axes=plt.gca())
    else:
        ax = sns.heatmap(comp, annot=False, linewidths=0.5, cmap="binary", square=True,
                         axes=plt.gca())

    plt.savefig("./plots/heatmap_%s.pdf" % cfg["id"])

    # better/alternative way, not argmaxing but averaging actual probabilities
    t = pd.DataFrame(preds[0])
    t["truth"] = truths.values
    comp2 = t.groupby("truth").apply(np.mean)
    del comp2["truth"]
    comp2 = comp2.T
    comp2.sort_index(ascending=False, inplace=True)

    plt.figure(figsize=(15, 12))
    plt.title(r"$P(\mathrm{prediction}\vert\mathrm{truth})$")

    if labels is not None:
        ax = sns.heatmap(comp2, annot=False, linewidths=0.5, cmap="binary",
                         xticklabels=labels, yticklabels=labels[::-1], square=True,
                         axes=plt.gca())
    else:
        ax = sns.heatmap(comp2, annot=False, linewidths=0.5, cmap="binary", square=True,
                         axes=plt.gca())

    plt.ylabel("prediction")

    plt.savefig("./plots/heatmap_exact_%s.pdf" % cfg["id"])


def PlotTrainingResult(logdf, cfg):

    fig = plt.figure()
    sfig = fig.add_axes([0.15, 0.11, 0.845, 0.78])

    sfig.set_xlabel(r"Epoch")
    sfig.set_ylabel(r"Accuracy")

    sfig.plot(logdf.epoch + 1, logdf.main_out_acc, ls="-", marker="o", color="k", label="Training accuracy")
    sfig.plot(logdf.epoch + 1, logdf.val_main_out_acc, ls="-", marker="s", color="r", label="Testing accuracy")

    plt.legend(loc="upper left")
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%i"))
    plt.ylim(0., 1.)

    plt.savefig("./plots/training_results_%s.pdf" % cfg["id"])


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
