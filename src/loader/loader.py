from src.constants import DATA_FOLDER
from src.util.filesystem import load_from_file


def clean_tweet(line):
    return line


def clean_test(line):
    return line


def load_data(sample=False):
    """

    :param sample: Wheither we should work on a small subset of the data
    :return:
    """
    pos_path = "train_pos.txt" if sample else "train_pos_full.txt"
    neg_path = "train_neg.txt" if sample else "train_neg_full.txt"
    test_path = "test_data.txt"

    return {
        'pos': load_from_file(DATA_FOLDER + pos_path, clean_fn=clean_tweet),
        'neg': load_from_file(DATA_FOLDER + neg_path, clean_fn=clean_tweet),
        'test': load_from_file(DATA_FOLDER + test_path, clean_fn=clean_test),
    }
