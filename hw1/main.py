import csv
import os
import pickle
import re

import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer

vectors = []

labels_dict = {
    'anger': 0,
    'fear': 1,
    'joy': 2,
    'love': 3,
    'sadness': 4,
    'surprise': 5
}
reverse_labels_dict = {v: k for k, v in labels_dict.items()}


# preprocessor.py
stop_words = {'against', 'wouldn', 'haven', 'ain', 'all', 'been', 'has', 'again', "couldn't", 'too', 'this', 'but',
              'your', 'because', "should've", 'themselves', 'do', 'll', 'yourself', "don't", 'hers', 'are', 'under',
              "haven't", 'after', 'theirs', 'shouldn', 'd', 'same', 'very', 'won', 'y', 'those', 'such', 'don', 'as',
              "needn't", 'them', 'just', "weren't", 'both', "that'll", "you'd", 'where', 'her', 'other', "mightn't",
              'and', 'm', 'we', 'have', 'only', 'itself', 'most', 'shan', 'weren', 'myself', 'had', 'ours', 'am',
              "aren't", 'wasn', "shan't", "wouldn't", "you're", 'whom', 'they', 'does', 'how', 'hadn', 'own', "hadn't",
              'before', 'him', 'below', 'while', 'ma', 'here', 'a', 'into', 'my', "won't", "you'll", 'between', 'its',
              'for', 'from', "she's", 'needn', "hasn't", 'herself', 'some', 'did', "isn't", 'up', 'or', 'our', 'of',
              'who', 'is', 'above', 'which', 'than', 'there', 'about', 'himself', 'so', 'i', 'having', 'to', 'doesn',
              "wasn't", 'that', 'these', 'yours', 'if', 'down', 'once', 'when', 'didn', 'will', 'was', 'on', 't',
              'their', 'ourselves', 'further', 's', "it's", 'then', 'mustn', 've', 'it', 'now', "you've", "doesn't",
              'were', 'yourselves', 'couldn', "didn't", 'an', 'mightn', 'out', 'you', "shouldn't", 'he', 'nor', 'why',
              'at', 'the', 'being', 're', 'doing', 'can', 'each', "mustn't", 'me', 'she', 'what', 'no', 'until', 'any',
              'more', 'aren', 'during', 'isn', 'not', 'in', 'o', 'his', 'off', 'few', 'be', 'over', 'hasn', 'by',
              'should', 'through', 'with'}
lemmatizer = WordNetLemmatizer()


def scrub(words):
    scrubbed_words = []
    for word in words:
        # remove trailing spaces
        word = word.strip()
        # remove non-ascii chars and digits
        word = re.sub(r'(\W|\d)', '', word)
        if word:
            scrubbed_words.append(word)
    return scrubbed_words


def remove_stopwords(words):
    return [word for word in words if word not in stop_words]


def lower(words):
    return [word.lower() for word in words]


# def stem(words):
#     return [stemmer.stem(word) for word in words]


def lemmatize(words):
    return [lemmatizer.lemmatize(word, pos='v') for word in words]


def _preprocess(sentence):
    words = sentence.split()
    words = scrub(words)
    words = remove_stopwords(words)
    words = lower(words)
    # words = stem(words)
    words = lemmatize(words)
    return ' '.join(words)


# ngram.py
class NGram:
    def __init__(self, n):
        self.n = n
        self.grams_dict = {}
        self.grams = []

    def fit_transform(self, series, k_best=None):
        for _, row in series.items():
            grams = self.text_to_grams(row)
            for gram in grams:
                self.grams_dict[gram] = self.grams_dict.get(gram, 0) + 1
        self.select_k_best(k_best)

        vectors = self.transform(series)
        return vectors

    def transform(self, series):
        # print('Transforming text to n-gram vectors.')
        vectors = np.array([np.zeros(len(self.grams))])
        total = series.shape[0]
        for i, row in series.items():
            # if i % 100 == 0:
            #     print(f'Processed {i}/{total} rows.')
            row_grams = self.text_to_grams(row)
            vector = np.array([1 if gram in row_grams else 0 for gram in self.grams])
            vectors = np.append(vectors, [vector], axis=0)
        vectors = np.delete(vectors, 0, axis=0)
        # vectors = pd.Series(vectors.tolist(), index=series.index)
        print('Transformation completed.')
        return vectors

    def select_k_best(self, k_best):
        total = len(self.grams_dict)
        # print(f'Selecting top {k_best} out of {total} features.')
        if k_best is None or k_best >= total:
            self.grams = list(self.grams_dict.keys())
        else:
            self.grams = sorted(self.grams_dict, key=self.grams_dict.get, reverse=True)[:k_best]

    def text_to_grams(self, row):
        words = row.split()
        length = len(words)
        if self.n == 1:
            row_grams = [tuple([words[i]]) for i in range(length)]
        else:
            row_grams = [tuple(words[i:i + self.n]) for i in range(length - self.n + 1)]
        return row_grams


# util.py
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


def transform_ngrams(data, grams=2, n_features=10000, vectorizer=None):
    return n_grams(data, grams=grams, k_best=n_features, gram=vectorizer)


def preprocess(df):
    print('Preprocessing...')
    data = df.copy()
    data['text'] = data['text'].apply(_preprocess)
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


def prepare_data(filepath, vectorizer=None):
    global vectors
    name = filename(filepath)
    method = 'unigram'
    n_features = 1000

    feature_path = f'models/{name}_{method}_{n_features}.pkl'
    vector_path = f'models/{name}_{method}_{n_features}_vectorizer.pkl'
    if os.path.exists(feature_path):
        data = pd.read_pickle(feature_path)
        with open(vector_path, 'rb') as f:
            vectors = pickle.load(f)
    else:
        data = pd.read_csv('train.csv')
        data, vectorizer = extract_features(data, method, n_features, vectorizer=vectorizer)
        vectors = vectorizer
        pd.to_pickle(data, feature_path)
        with open(vector_path, 'wb') as f:
            pickle.dump(vectorizer, f)
    X = data.iloc[:, 3:]
    Y = pd.get_dummies(data['emotions'])
    return pd.concat((X, Y), axis=1)


def prepare_test_data():
    global vectors
    test_df = pd.read_csv('test.csv')
    test_df, _ = extract_features(test_df, 'unigram', None, vectorizer=vectors)
    test_df = test_df.iloc[:, 2:].to_numpy()
    return test_df


def filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def split_data_labels(*args):
    splits = []
    for _data in args:
        y = _data.iloc[:, -6:].to_numpy()
        x = _data.drop(columns=_data.columns[-6:], axis=1).to_numpy()
        splits.append((x, y))
    return splits


def split_data(df):
    test = df.sample(frac=0.2, random_state=46)
    train = df.drop(test.index)
    splits = split_data_labels(test, train)
    train_data, train_labels, test_data, test_labels = splits[0][0], splits[0][1], splits[1][0], splits[1][1]
    return train_data, train_labels, test_data, test_labels


# lr.py
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(X):
    ps = np.empty(X.shape)
    for i in range(X.shape[0]):
        ps[i, :] = np.exp(X[i, :])
        ps[i, :] /= np.sum(ps[i, :])
    return ps


# noinspection PyPep8Naming
class Logistic:
    def __init__(self, data, labels, learning_rate=0.1, max_epochs=200, lam=0.001):
        self.X = data
        self.Y = labels

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.lam = lam

        self.weights = np.random.rand(self.X.shape[1], self.Y.shape[1])

    def gradient(self, lam):
        z = -self.X.dot(self.weights)
        p = softmax(z)
        return (self.X.T.dot(self.Y - p)) / self.X.shape[0]

    def fit(self):
        for epoch in range(self.max_epochs):
            self.weights -= self.learning_rate * self.gradient(self.lam) + self.lam * self.weights

    def predict(self, X):
        z = -np.dot(X, self.weights)
        p = softmax(z)
        predictions = np.argmax(p, axis=1)
        return predictions


# nn.py
def cross_entropy(y, y_pred):
    y_pred += 1e-7
    return -np.sum(y * np.log(y_pred)) / y.shape[0]


def mse(y, y_pred):
    return np.mean(np.square(y - y_pred))


def loss_fn(y, y_pred):
    return cross_entropy(y, y_pred)


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, train, labels, alpha, lam, epochs, hidden_neurons):
        self.train = train
        self.labels = labels
        self.alpha = alpha
        self.lam = lam
        self.epochs = epochs

        self.hidden_neurons = hidden_neurons
        self.w1 = np.random.rand(self.train.shape[1], self.hidden_neurons)
        self.b1 = np.zeros(self.hidden_neurons)

        self.w2 = np.random.rand(self.hidden_neurons, self.labels.shape[1])
        self.b2 = np.zeros(self.labels.shape[1])

    def fit(self):
        batch_size = 200
        x = self.train
        y = self.labels
        for epoch in range(self.epochs):
            loss = 0
            for i in range(0, len(self.train), batch_size):
                x = self.train[i: i + batch_size]
                y = self.labels[i: i + batch_size]

                # Forward pass
                z1 = np.dot(x, self.w1) + self.b1
                a1 = sigmoid(z1)

                z2 = np.dot(a1, self.w2) + self.b2
                a2 = softmax(z2)

                # loss += loss_fn(y, a2)

                # Backward pass
                delta2 = a2 - y
                d_w2 = np.dot(a1.T, delta2)
                d_b2 = np.sum(delta2, axis=0)

                delta1 = np.dot(delta2, self.w2.T) * sigmoid_derivative(a1)
                d_w1 = np.dot(x.T, delta1)
                d_b1 = np.sum(delta1, axis=0)

                # Update weights and biases
                self.w2 -= self.alpha * d_w2 + (self.lam * self.w2)
                self.b2 -= self.alpha * d_b2 + (self.lam * self.b2)

                self.w1 -= self.alpha * d_w1 + (self.lam * self.w1)
                self.b1 -= self.alpha * d_b1 + (self.lam * self.b1)

                # print("Epoch: {}, Loss: {}".format(epoch, loss))

    def predict(self, inputs):
        # Forward pass
        z1 = np.dot(inputs, self.w1) + self.b1
        a1 = sigmoid(z1)

        z2 = np.dot(a1, self.w2) + self.b2
        a2 = softmax(z2)
        predictions = np.argmax(a2, axis=1)
        return predictions


# main.py
def write_predictions(predictions, file_name):
    fieldnames = ['id', 'text', 'emotions']
    results = []
    with open('test.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        next(reader)
        for i, row in enumerate(reader):
            result = {
                'id': row['id'],
                'text': row['text'],
                'emotions': reverse_labels_dict[predictions[i]]
            }
            results.append(result)
    with open(file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def predict_test_data(lr, file_name):
    predictions = lr.predict(test_data)
    write_predictions(predictions, file_name)


def LR():
    alpha = 0.5
    lam = 0.001
    epoch = 500

    train_X, train_Y, val_X, val_Y = split_data(data)
    lr = Logistic(train_X, train_Y, alpha, epoch, lam)
    lr.fit()
    predictions = lr.predict(val_X)
    if not len(val_Y.shape) == 1:
        val_Y = np.argmax(val_Y, axis=1)
    accuracy = np.mean(predictions == val_Y)
    print(f'Accuracy: {accuracy}')

    predict_test_data(lr, file_name='test_lg.csv')


def NN():
    alpha = 0.01
    epoch = 200
    lam = 0.0001
    hidden_neurons = 37

    # your Multi-layer Neural Network
    train_data, train_labels, val_data, val_labels = split_data(data)

    neural_net = NeuralNetwork(train_data, train_labels, alpha=alpha, lam=lam, epochs=epoch,
                               hidden_neurons=hidden_neurons)
    neural_net.fit()
    predictions = neural_net.predict(val_data)
    if not len(val_labels.shape) == 1:
        val_labels = np.argmax(val_labels, axis=1)
    accuracy = np.mean(predictions == val_labels)
    print(f'Accuracy: {accuracy}')

    predict_test_data(neural_net, file_name='test_nn.csv')


data = prepare_data('train.csv')
test_data = prepare_test_data()

if __name__ == '__main__':
    print("..................Beginning of Logistic Regression................")
    LR()
    print("..................End of Logistic Regression................")

    print("------------------------------------------------")

    print("..................Beginning of Neural Network................")
    NN()
    print("..................End of Neural Network................")
