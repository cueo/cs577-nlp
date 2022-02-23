import os
import pickle

import numpy as np
import pandas as pd

import preprocessor as p
from const import labels_dict
from ngram import NGram


def n_grams(df, grams, k_best=None, gram=None):
    # print(f'Building {grams}-grams with {k_best} features.')
    if gram is None:
        gram = NGram(grams)
        x = gram.fit_transform(df['text'], k_best=k_best)
    else:
        x = gram.transform(df['text'])

    data = df.copy()
    data = data.join(pd.DataFrame(x, columns=[f'f_{i}' for i in range(x.shape[1])]))
    # data['vector'] = data['vector'].apply(np.array)
    return data, gram


def save(model, filename):
    with open(f'models/{filename}.pkl', 'wb') as f:
        pickle.dump(model, f)


def load(filename):
    with open(f'models/{filename}.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def transform_ngrams(data, grams=2, n_features=10000, vectorizer=None):
    return n_grams(data, grams=grams, k_best=n_features, gram=vectorizer)


def preprocess(df):
    print('Preprocessing...')
    data = df.copy()
    data['text'] = data['text'].apply(p.preprocess)
    return data


def extract_features(data, method, n_features, vectorizer=None, train=False):
    if method == 'unigram':
        grams = 1
    elif method == 'trigram':
        grams = 3
    else:
        grams = 2
    if train:
        filepath = f'models/data_{method}_{n_features}.pkl'
        vec_path = f'models/data_vectorizer_{method}_{n_features}.pkl'
        if os.path.exists(filepath):
            all_data = pd.read_pickle(filepath)
            with open(vec_path, 'rb') as f:
                vectorizer = pickle.load(f)
            if len(all_data) != len(data) or 'vector' not in data.columns:
                data = all_data.iloc[data.index]
        else:
            data, vectorizer = extract(data, grams, n_features, vectorizer)
            data.to_pickle(filepath)
            with open(vec_path, 'wb') as f:
                pickle.dump(vectorizer, f)
    else:
        data, vectorizer = extract(data, grams, n_features, vectorizer)
    return data, vectorizer


def extract(data, grams, n_features, vectorizer):
    data = preprocess(data)
    data, vectorizer = transform_ngrams(data, grams, n_features, vectorizer)
    return data, vectorizer


def split_data_labels(data, test, train):
    train_labels = train.iloc[:, -6:].to_numpy()
    train_data = train.drop(columns=data.columns[-6:], axis=1).to_numpy()
    test_labels = test.iloc[:, -6:].to_numpy()
    test_data = test.drop(columns=data.columns[-6:], axis=1).to_numpy()
    return train_data, train_labels, test_data, test_labels


def split_data(data):
    test = data.sample(frac=0.2, random_state=46)
    train = data.drop(test.index)
    train_data, train_labels, test_data, test_labels = split_data_labels(data, test, train)
    return train_data, train_labels, test_data, test_labels


def cv_split_data(data, fold_num):
    length = len(data)
    start = int(fold_num * 0.2 * length)
    end = int((fold_num + 1) * 0.2 * length)
    train = pd.concat([data[:start], data[end:]])
    test = data[start:end]
    test_data, test_labels, train_data, train_labels = split_data_labels(data, test, train)
    return train_data, train_labels, test_data, test_labels


def accuracy(labels, predictions):
    return np.mean(labels == predictions)


def feature_extraction(data, method, n_features, train=False):
    data, _ = extract_features(data, method, n_features, train=train)
    return data


def fn(x):
    return labels_dict[x]


def prepare_data(filepath):
    name = filename(filepath)
    method = 'unigram'
    n_features = 1000

    feature_path = f'models/{name}_{method}_{n_features}.pkl'
    if os.path.exists(feature_path):
        data = pd.read_pickle(feature_path)
    else:
        data = pd.read_csv('train.csv')
        data, _ = extract_features(data, method, n_features)
        pd.to_pickle(data, feature_path)
    X = data.iloc[:, 3:]
    Y = pd.get_dummies(data['emotions'])
    return pd.concat((X, Y), axis=1)


def filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def encode_onehot(a):
    """
    Create one-hot encoding of the labels with rows as labels and columns as data rows.
    Args:
        a: labels

    Returns:
        one-hot encoding of the labels
    """
    a_onehot = np.zeros((a.size, len(labels_dict)))
    for x in a.iteritems():
        a_onehot[x[0], x[1]] = 1
    return a_onehot
