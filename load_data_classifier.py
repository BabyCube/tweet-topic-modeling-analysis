import re
import io
import numpy as np
import pandas
import json
import random
import gluonnlp as nlp
from sklearn.preprocessing import OneHotEncoder


def load_dataset(tweets_file, upper, lower, max_length=32):
    """
    load dataset and split the data from one file into three

    Inputs: one JSON file with each line representing a tweet object
    Outputs: vocabulary (with attached embedding), training, validation and test datasets ready for neural net training
    """
    file = open(tweets_file, 'r')
    val_array = []
    test_array = []
    train_array = []

    for tweet in file:
        tweet_object = json.loads(tweet.strip())
        split_key = random.uniform(0, 1)
        if split_key > 0.9:
            test_array.append([tweet_object["SIA_sentiment"], tweet_object["text"]])
        elif split_key < 0.1:
            val_array.append([tweet_object["SIA_sentiment"], tweet_object["text"]])
        else:
            train_array.append([tweet_object["SIA_sentiment"], tweet_object["text"]])
    print ("test array length: ", len(test_array), "validation array length: ",
           len(val_array), "train array length: ",len(train_array))
    vocab = build_vocabulary(train_array, val_array, test_array)

    train_dataset = preprocess_dataset(train_array, vocab, upper, lower, max_length)
    val_dataset = preprocess_dataset(val_array, vocab, upper, lower, max_length)
    test_dataset = preprocess_dataset(test_array, vocab, upper, lower, max_length)

    return vocab, train_dataset, val_dataset, test_dataset


def build_vocabulary(tr_array, val_array, tst_array):
    """
    Inputs: arrays representing the training, validation and test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    all_tokens = []
    tokenize = nlp.data.BERTBasicTokenizer(lower=True)
    for i, text in enumerate(tr_array):
        tokens = tokenize(text[1])
        all_tokens.extend(tokens)
    ### TODO: ... same for val_array and tst_array
    for i, text in enumerate(val_array):
        tokens = tokenize(text[1])
        all_tokens.extend(tokens)

    for i, text in enumerate(tst_array):
        tokens = tokenize(text[1])
        all_tokens.extend(tokens)

    counter = nlp.data.count_tokens(all_tokens)
    vocab = nlp.Vocab(counter)
    print(vocab)
    return vocab


def _preprocess(x, vocab, max_len, upper, lower):
    """
    Inputs: data instance x (tokenized), vocabulary, maximum length of input (in tokens)
    Outputs: data mapped to token IDs, with corresponding label
    """
    tokenize = nlp.data.BERTBasicTokenizer(lower=True)
    score, text_tokens = x
    data = vocab[tokenize(text_tokens)]
    ## TODO: data = ... ensure does not exceed max_len; pad remaining tokens with pad_id (0)
    if len(data) < max_len:
        pad_length = max_len - len(data)
        pad_data = []
        while pad_length > 0:
            pad_data.append(0) # pad to 0 if exceed max length!
            pad_length = pad_length - 1
        data = data + pad_data
    elif len(data) > max_len:
        data = data[:max_len]
    # Encode label from upper/lower bound
    # -1 means negative emotion while 1 means positive emotion
    if score > upper:
        label = 1
    elif score < lower:
        label = -1
    else:
        label = 0
    # print(label, data)
    return label, data


def preprocess_dataset(dataset, vocab, upper, lower, max_len):
    preprocessed_dataset = [_preprocess(x, vocab, max_len, upper, lower) for i, x in enumerate(dataset)]
    return preprocessed_dataset


class BasicTransform(object):
    """
    This is a callable object used by the transform method for a dataset. It will be
    called during data loading/iteration.

    Parameters
    ----------
    labels : list string
        List of the valid strings for classification labels
    max_len : int, default 64
        Maximum sequence length - longer seqs will be truncated and shorter ones padded

    """

    def __init__(self, labels, max_len=64):
        self._max_seq_length = max_len
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i

    def __call__(self, label, data):
        label_id = self._label_map[label]
        padded_data = data + [0] * (self._max_seq_length - len(data))
        return mx.nd.array(padded_data, dtype='inlt32'), mx.nd.array([label_id], dtype='int32')

