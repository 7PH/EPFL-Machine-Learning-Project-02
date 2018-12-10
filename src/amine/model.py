from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer


def tokenize(df_train, df_test):
    # At this stage, we train keras tokenizer
    tokenizer = Tokenizer(split=' ')
    tokenizer.fit_on_texts(df_train['tweet'].values)
    dict_token = tokenizer.word_index  # To see the dicstionary of words kept

    # We store the number of features of the neural net since we havent' fixed it in tokenization
    nb_features = len(dict_token)

    # Converting tweets to sequences and then padding them
    train_sequences = tokenizer.texts_to_sequences(df_train['tweet'].values)
    train_sequences_pad = sequence.pad_sequences(train_sequences, maxlen=30)

    test_sequences = tokenizer.texts_to_sequences(df_test['tweet'].values)
    test_sequences_pad = sequence.pad_sequences(test_sequences, maxlen=30)

    return {
        'nb_features': nb_features,
        'train_sequences_pad': train_sequences_pad,
        'test_sequences_pad': test_sequences_pad,
    }


def get_model(nb_features, train_sequences_pad):
    embed_dim = 50
    lstm_out = 100
    model = Sequential()
    model.add(Embedding(nb_features + 1, embed_dim, input_length=train_sequences_pad.shape[1]))
    # model.add(Dropout(0.5))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
