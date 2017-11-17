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

    PrepareData(cfg)
    data = cfg.data
    qs = data["meta"]
    conn = data["dbconn"]

    for fit in cfg.fits:
        print "Working on fit of type %s with name %s" % (fit["type"], fit["name"])

        assert "embed_path" in fit, "Embedding input is not defined! This is currently required!"

        if not os.path.exists(fit_nn["embed_out"]):
            ConvertToGensimFile(fit_nn["embed_path"], fit_nn["embed_out"])

        gmod = KeyedVectors.load_word2vec_format(fit_nn["embed_out"], binary=False)

        assert "labelfct" in fit, "Necessary to provide label function!"

        # calculating labels
        qs["label"] = fit["labelfct"](qs)
        nsample = fit.get("nsample", 100000)

        if fit.get("uniform", True):

            print "Selecting a sample of %i posts randomly and uniformly." % nsample
            qssel = SelectUniformlyFromColumn(qs, "label", n=nsample)

        else:
            print "Selecting a sample of %i posts randomly." % nsample
            qssel = qs.sample(nsample)

        qstrain = qssel.iloc[:int(0.8 * qssel.shape[0])]
        qstest = qssel.iloc[int(0.8 * qssel.shape[0]):]
        print "Length of the training set:", len(qstrain)
        print "Length of the testing set:", len(qstest)


posts_train = GetDBPosts(qstrain.Id.values, conn)
posts_test = GetDBPosts(qstest.Id.values, conn)
conn.close()


titles_train = np.squeeze(qstrain.Title.values)
titles_test = np.squeeze(qstest.Title.values)


max_features = 50000

word_tokenizer = Tokenizer(max_features)
word_tokenizer.fit_on_texts(posts_train)


# In[11]:


# actual tokenization using the tokenizer from above
posts_train_tf = word_tokenizer.texts_to_sequences(posts_train)
posts_test_tf = word_tokenizer.texts_to_sequences(posts_test)

# padding to a maximal question length for all questions
maxlen_posts = 1000
posts_train_tf = pad_sequences(posts_train_tf, maxlen=maxlen_posts, padding="post", truncating="post")
posts_test_tf = pad_sequences(posts_test_tf, maxlen=maxlen_posts, padding="post", truncating="post")

print(posts_train_tf[0])


# In[27]:


titles_train_tf = word_tokenizer.texts_to_sequences(titles_train)
titles_test_tf = word_tokenizer.texts_to_sequences(titles_test)

# padding to a maximal title length
maxlen_titles = 50
titles_train_tf = pad_sequences(titles_train_tf, maxlen=maxlen_titles, padding="post", truncating="post")
titles_test_tf = pad_sequences(titles_test_tf, maxlen=maxlen_titles, padding="post", truncating="post")

print(titles_train_tf[0])


# In[13]:


# setting up weights matrix for embedding in keras
weights_matrix = np.zeros((max_features + 1, embed_dim))

for word, i in word_tokenizer.word_index.items():

    if i > max_features:
        continue
    try:
#         embedding_vector = embedding_vectors.get(word)
        embedding_vector = gensimmodel.word_vec(word)
        if embedding_vector is not None:
            weights_matrix[i] = embedding_vector
    except:
        weights_matrix[i] = np.zeros(embed_dim)


# In[14]:


batch_size = 100
epochs = 20
split = 0.2


# In[15]:


# setting up posts branch for modeling
posts_input = Input(shape=(maxlen_posts,), name="posts_input")
posts_embedding = Embedding(max_features + 1, embed_dim, weights=[weights_matrix])(posts_input)
posts_pooling = GlobalAveragePooling1D()(posts_embedding)

aux_output = Dense(1, activation="sigmoid", name="aux_out")(posts_pooling)


# In[18]:


# setting up posts branch for modeling
titles_input = Input(shape=(maxlen_titles,), name="titles_input")
titles_embedding = Embedding(max_features + 1, embed_dim, weights=[weights_matrix])(titles_input)
titles_pooling = GlobalAveragePooling1D()(titles_embedding)

aux_output2 = Dense(1, activation="sigmoid", name="aux_out2")(titles_pooling)


# In[19]:


# adding embeddings for other features
relcols = ["BodyNCodes", "BodyNQMarks", "BodySize", "titlelen", "nwords", "ordersum", "ordermean", "orderstd", "ratio"]
# todo: extend here to actually add all needed embeddings in dynamic way

meta_embedding_dims = 64

hours_input = Input(shape=(1,), name="hours_input")
hours_embedding = Embedding(24, meta_embedding_dims)(hours_input)
hours_reshape = Reshape((meta_embedding_dims,))(hours_embedding)

dayofweeks_input = Input(shape=(1,), name="dayofweeks_input")
dayofweeks_embedding = Embedding(7, meta_embedding_dims)(dayofweeks_input)
dayofweeks_reshape = Reshape((meta_embedding_dims,))(dayofweeks_embedding)

dayofyears_input = Input(shape=(1,), name="dayofyears_input")
dayofyears_embedding = Embedding(366, meta_embedding_dims)(dayofyears_input)
dayofyears_reshape = Reshape((meta_embedding_dims,))(dayofyears_embedding)


# In[20]:


# connecting the different embeddings
merged = concatenate([posts_pooling, titles_pooling, hours_reshape, dayofweeks_reshape, dayofyears_reshape])

hidden_1 = Dense(256, activation="relu")(merged)
hidden_1 = BatchNormalization()(hidden_1)

main_output = Dense(1, activation="sigmoid", name="main_out")(hidden_1)


# In[21]:


model = Model(inputs=[posts_input,
                      titles_input,
                      hours_input,
                      dayofweeks_input,
                      dayofyears_input], outputs=[main_output, aux_output, aux_output2])

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"],
              loss_weights=[1, 0.2, 0.2])

model.summary()


# In[56]:


plot_model(model, to_file='./plots/keras_model.png')
plot_model(model, to_file='./plots/model_shapes.png', show_shapes=True)


# In[22]:


print np.sum(qstrain[label]), qstrain[label].shape
print 1 - np.sum(qstrain[label]) * 1. / qstrain[label].shape[0]
print 1 - np.sum(qstest[label]) * 1. / qstest[label].shape[0]
print(1 - np.mean(qstrain[label][:(int(posts_train_tf.shape[0] * split))]))
print(1 - np.mean(qstest[label][:(int(posts_test_tf.shape[0] * split))]))


# In[24]:


csv_logger = CSVLogger('training.csv')


# In[25]:


# fitting :)
model.fit([posts_train_tf, titles_train_tf, qstrain.dayhour.values, qstrain.weekday.values, qstrain.day.values],
          [qstrain[label], qstrain[label], qstrain[label]],
          batch_size=batch_size,
          epochs=5,
          validation_split=split, callbacks=[csv_logger])


# In[28]:


a = model.evaluate(x=[posts_test_tf, titles_test_tf, qstest.dayhour.values, qstest.weekday.values, qstest.day.values],
                   y=[qstest[label], qstest[label], qstest[label]])


# In[33]:


model.save("keras_ispython.nnmodel")
model.save_weights("keras_ispython_weights.nnmodel")


# In[29]:


print a


# In[49]:


preds = a[0]


# In[60]:


preds_bin = np.around(preds).T[0]


# In[64]:


print np.sum(preds_bin == label)
print len(preds_bin)




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
