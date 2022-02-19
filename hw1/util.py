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


def split_data(data, labels):
    # train_data, test_data = all_data.iloc[:int(len(all_data) * 0.8), :], all_data.iloc[int(len(all_data) * 0.8):, :]
    #     return train_data, test_data
    train_data, test_data = data.iloc[:int(len(data) * 0.8), :], data.iloc[int(len(data) * 0.8):, :]
    train_labels, test_labels = labels.iloc[:int(len(labels) * 0.8)], labels.iloc[int(len(labels) * 0.8):]
    return train_data, train_labels, test_data, test_labels


def cv_split_data(data, labels, fold_num):
    assert len(data) == len(labels)
    length = len(data)
    start = int(fold_num * 0.2 * length)
    end = int((fold_num + 1) * 0.2 * length)
    train_data = pd.concat([data[:start], data[end:]])
    train_labels = pd.concat([labels[:start], labels[end:]])
    test_data = data[start:end]
    test_labels = labels[start:end]
    return train_data, train_labels, test_data, test_labels


def accuracy(labels, predictions):
    return np.mean(labels == predictions)


def feature_extraction(data, method, n_features, train=False):
    data, _ = extract_features(data, method, n_features, train=train)
    return data


def f(x):
    return labels_dict[x]


def prepare_data(filepath):
    name = filename(filepath)
    feature_path = f'models/{name}.pkl'
    if os.path.exists(feature_path):
        data = pd.read_pickle(feature_path)
    else:
        method = 'unigram'
        n_features = 1000
        data = pd.read_csv('train.csv')
        data, _ = extract_features(data, method, n_features)
        pd.to_pickle(data, feature_path)
    X = data.iloc[:, 3:]
    Y = data['emotions'].apply(f)
    return X, Y


def filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]
