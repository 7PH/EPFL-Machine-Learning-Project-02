import pandas as pd
import numpy as np
from src.constants import DATA_FOLDER
from src.util.filesystem import load_from_file


def clean_tweet(line):
    return line


def clean_test(line):
    return line


def load_data(sample=False, shuffle=True):
    """

    :param sample: Whether we should work on a small subset of the data
    :param shuffle: Shuffle the datasets?
    :return: dict(test: .., train: ..)
    """
    pos_path = "train_pos.txt" if sample else "train_pos_full.txt"
    neg_path = "train_neg.txt" if sample else "train_neg_full.txt"
    test_path = "test_data.txt"

    df_pos = pd.DataFrame(data={'tweet': load_from_file(DATA_FOLDER + pos_path, clean_fn=clean_tweet)})
    df_pos['label'] = 0
    df_neg = pd.DataFrame(data={'tweet': load_from_file(DATA_FOLDER + neg_path, clean_fn=clean_tweet)})
    df_neg['label'] = 1
    df_test = pd.DataFrame(data={'tweet': load_from_file(DATA_FOLDER + test_path, clean_fn=clean_test)})
    df_test['label'] = -1
    df_test = df_test.reset_index(drop=True)

    df_train = pd.concat([df_neg, df_pos])
    if shuffle:
        df_train = df_train.iloc[np.random.permutation(len(df_train))]
    df_train = df_train.reset_index(drop=True)

    return {
        'test': df_test,
        'train': df_train
    }
