import numpy as np
import os
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer

from src.constants import DATA_FOLDER


def tokenize(df_train, df_test):
    # At this stage, we train keras tokenizer
    tokenizer = Tokenizer(split=' ')
    tokenizer.fit_on_texts(df_train['tweet'].values)
    word_index = tokenizer.word_index  # To see the dicstionary of words kept

    # We store the number of features of the neural net since we havent' fixed it in tokenization
    nb_features = len(word_index)

    # Converting tweets to sequences and then padding them
    train_sequences = tokenizer.texts_to_sequences(df_train['tweet'].values)
    train_sequences_pad = sequence.pad_sequences(train_sequences, maxlen=30)

    test_sequences = tokenizer.texts_to_sequences(df_test['tweet'].values)
    test_sequences_pad = sequence.pad_sequences(test_sequences, maxlen=30)

    return {
        'nb_features': nb_features,
        'train': train_sequences_pad,
        'test': test_sequences_pad,
        'word_index': word_index
    }


def get_embedding_matrix(word_index, embed_dim):
    embeddings_index = {}
    f = open(os.path.join(DATA_FOLDER + 'glove.twitter.27B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def get_model_cnn(nb_features, train_sequences_pad, embedding_matrix, lstm_out, embed_dim):
    model = Sequential()
    emb = Embedding(
        nb_features + 1,
        embed_dim,
        weights=[embedding_matrix],
        input_length=train_sequences_pad.shape[1]
    )
    emb.trainable = True
    model.add(emb)
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(lstm_out))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def get_model(nb_features, train_sequences_pad, embedding_matrix, lstm_out, embed_dim):
    model = Sequential()
    emb = Embedding(
        nb_features + 1,
        embed_dim,
        weights=[embedding_matrix],
        input_length=train_sequences_pad.shape[1]
    )
    emb.trainable = True
    model.add(emb)
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
