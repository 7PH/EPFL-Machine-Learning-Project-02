import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from src.constants import DATA_FOLDER
from src.util.filesystem import load_from_file

nltk.download('stopwords')
stop = stopwords.words('english')


def decontracted(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def clean_tweet_glove(tweet):
    tweet = " " + tweet + " "
    glove_regex = {
        r",": " , ",
        r"#": " # ",
        r"\s{2,}": " ",
        r";": " ; ",
        r":": " : ",
        r"\'s": " \'s",
        r"\'ve": " have",
        r"n\'t": " not",
        r"\'re": " \'re",
        r"\'d": " \'d",
        r" ([!?.])\s?[!?.]{1,} ": r" \1 <REPEAT> ",
        r" [-+]?[.\d ]*[\d]+[:,.\d]* ": " <NUMBER> ",
        r" (\w+?)(\w)(\2{2,}) ": r" \1\2 <ELONG> "
    }

    for gregex in glove_regex:
        tweet = re.sub(gregex, glove_regex[gregex], tweet)

    tweet = tweet.strip().split(" ")  # remove the white space at the beggining and the end of the tweet
    # and split the tweet by white space
    tweet = [i for i in tweet if i not in stop]
    tweet = ["<HASHTAG>" if i == '#' else i for i in tweet]

    return " ".join([word for word in tweet])


def clean_tweet_casual(tweet):
    tweet = decontracted(tweet)

    glove_regexs = {}
    glove_regexs[r'[+-]?([0-9]+)([\.,]?([0-9]*))'] = r' <number> '
    glove_regexs[r'(\?{2,})'] = r' \?<multiquestionmark> '
    glove_regexs[r'(\!{2,})'] = r' \!<multiexclampoint> '
    glove_regexs[r'([A-Z]{3,})'] = r'<allcap> \1'
    glove_regexs[r'#([a-zA-Z-_]+)'] = r'<hashtag> \1'
    glove_regexs[r'[ ]{2,}'] = r' '
    glove_regexs[r'([a-zA-Z]+?)([a-zA-Z])\2{2,}([a-zA-Z]+)*'] = r'\1\2\3 <elong>'

    for gregex in glove_regexs:
        tweet = re.sub(gregex, glove_regexs[gregex], tweet)

    tweet = tweet.lower().strip().split(" ")
    tweet = [i for i in tweet if i not in stop]

    return " ".join(tweet)


def clean_tweet_none(tweet):
    return tweet


def load_data(sample=False, shuffle=True, clean_fn=clean_tweet_none):
    """

    :param clean_fn:
    :param sample: Whether we should work on a small subset of the data
    :param shuffle: Shuffle the datasets?
    :return: dict(test: .., train: ..)
    """
    pos_path = "train_pos.txt" if sample else "train_pos_full.txt"
    neg_path = "train_neg.txt" if sample else "train_neg_full.txt"
    test_path = "test_data.txt"

    df_pos = pd.DataFrame(data={'tweet': load_from_file(DATA_FOLDER + pos_path, clean_fn=clean_fn)})
    df_pos['label'] = 1
    df_neg = pd.DataFrame(data={'tweet': load_from_file(DATA_FOLDER + neg_path, clean_fn=clean_fn)})
    df_neg['label'] = 0
    df_test = pd.DataFrame(data={'tweet': load_from_file(DATA_FOLDER + test_path, clean_fn=clean_fn)})
    df_test['id'] = df_test['tweet'].map(lambda x: x.split(',')[0])
    df_test['tweet'] = df_test['tweet'].map(lambda x: ','.join(x.split(',')[1:]))

    df_train = pd.concat([df_neg, df_pos])
    if shuffle:
        df_train = df_train.iloc[np.random.permutation(len(df_train))]
    df_train = df_train.reset_index(drop=True)

    return {
        'test': df_test,
        'train': df_train
    }
