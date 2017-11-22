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


# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
# adapted slightly
def TextCleansing(text, remove_stopwords=True, stem_words=False):
    """ Clean the text, with the option to remove stopwords and to stem words. """
    try:
        from spacy.lang.en import STOP_WORDS as STOPWORDS
    except:
        from spacy.en import STOPWORDS

    from nltk.stem import SnowballStemmer

    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        text = [w for w in text if w not in STOPWORDS]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer("english")
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return text


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
            qs["label"] = fitcfg["labelfct"](qs)
            nsample = fitcfg.get("nsample", 100000)

            if fitcfg.get("uniform", True):

                print "Selecting a sample of %i posts uniformly and randomly within each group." % nsample
                qssel = SelectUniformlyFromColumn(qs, "label", n=nsample)

            else:
                print "Selecting a sample of %i posts randomly." % nsample
                qssel = qs.sample(nsample)

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
                posts_train = GetDBPosts(qstrain.Id.values, conn)
                posts_test = GetDBPosts(qstest.Id.values, conn)

                if fitcfg.get("clean", False):
                    print "Cleaning posts..."
                    posts_train = [TextCleansing(p) for p in posts_train]
                    posts_test = [TextCleansing(p) for p in posts_test]
                else:
                    print "Warning! Posts are not cleaned! (stop-words, lemmatization etc)"

                print "Fitting tokenizer..."
                word_tokenizer = Tokenizer(fitcfg["nfeatures"])
                word_tokenizer.fit_on_texts(posts_train)

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

                if not word_tokenizer:
                    print "Building tokenizer on titles."
                    word_tokenizer = Tokenizer(fit["nfeatures"])
                    word_tokenizer.fitcfg_on_texts(titles_train)

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

            if fitcfg.get("cnn", False):
                print "Using CNN layer in network, please check options for filter and kernel size."
                merged = Conv1D(250, 3, padding="valid",
                                activation="relu", strides=1)(merged)
                merged = GlobalMaxPooling1D()(merged)

            # if fitcfg.get("LSTM", False):
                # model.add(LSTM(int(document_max_num_words*1.5), input_shape=(document_max_num_words, num_features)))
                # model.add(Dropout(0.3))

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

            if fitcfg.get("save", False):

                model.save("./models/keras_full_%s.keras" % fitcfg["id"])
                model.save_weights("./models/keras_weights_%s.keras" % fitcfg["id"])

                dill.dump(test_preds, open("./models/test_predictions_%s.dill" % fitcfg["id"], "w"))
                dill.dump(test_truths, open("./models/test_truths_%s.dill" % fitcfg["id"], "w"))
                dill.dump(qstest, open("./models/test_df_%s.dill" % fitcfg["id"], "w"))

        else:
            print "Using cached results!"
            from keras.models import load_model

            train_log = pd.read_csv("./logging/training_%s.csv" % fitcfg["id"])

            model = load_model("./models/keras_full_%s.keras" % fitcfg["id"])
            test_truths = dill.load(open("./models/test_truths_%s.dill" % fitcfg["id"], "r"))
            test_preds = dill.load(open("./models/test_preds_%s.dill" % fitcfg["id"], "r"))
            # test_df = dill.load(open("./models/test_df_%s.dill" % fitcfg["id"], "r"))

        # embed()

        if fitcfg.get("plots", True):

            print "Making a few plots..."
            PlotTrainingResult(train_log, fitcfg)
            PlotConfusionMatrix(test_truths, test_preds[0], fitcfg, labels=fitcfg.get("grouplabels", None))
            # PlotConfusionMatrix(test_truths, test_preds[0], fitcfg)

            if True:
                # plt.clear()
                plt.figure()
                cfg.mostcommon_tags.set_index("tags").head(20).plot.bar(ax=plt.gca())
                plt.ylabel("Number of tagged questions")
                plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d"))
                plt.savefig("./plots/hist_mostcommontags.pdf")


def PlotConfusionMatrix(truths, preds, cfg, labels=None):

    import matplotlib.cm as cm

    preds_bin = np.argmax(preds, axis=1)

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
        if isinstance(labels, list):
            labels = [r"%s" % l for l in labels]
        ax = sns.heatmap(comp, annot=False, linewidths=0.5, cmap="jet",
                         xticklabels=labels, yticklabels=labels[::-1], square=True)
    else:
        ax = sns.heatmap(comp, annot=False, linewidths=0.5, cmap="jet", square=True)

    plt.savefig("./plots/heatmap_%s.pdf" % cfg["id"])


def PlotTrainingResult(logdf, cfg):

    fig = plt.figure()
    sfig = fig.add_axes([0.15, 0.11, 0.845, 0.78])

    sfig.set_xlabel(r"Epoch")
    sfig.set_ylabel(r"Accuracy")

    sfig.plot(logdf.epoch + 1, logdf.main_out_acc, ls="-", marker="o", color="k", label="Training accuracy")
    sfig.plot(logdf.epoch + 1, logdf.val_main_out_acc, ls="-", marker="s", color="r", label="Testing accuracy")

    plt.legend(loc="upper left")
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
