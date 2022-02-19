import numpy as np
from scipy.special import softmax

from const import labels_dict


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


class Logistic:
    # Suppress PyArgumentNames
    def __init__(self, data, labels):
        self.X = data
        self.Y = encode_onehot(labels)
        self.weights = np.zeros((self.X.shape[1], self.Y.shape[1]))
        # self.bias = np.zeros(self.Y.shape[0])

        # self.method = None
        # self.n_features = 5000
        # self.vectorizer = None

    # def feature_extraction(self, data, train=False):
    #     data, vectorizer = util.extract_features(data, self.method, self.n_features, self.vectorizer, train)
    #     self.vectorizer = vectorizer
    #     return data

    def logistic_loss(self, predicted_label, true_label):
        pass

    def regularizer(self, lam=0.03):
        return self.weights * lam

    def gradient(self, lam):
        z = -self.X.dot(self.weights)
        # TODO: implement softmax
        p = softmax(z, axis=1)
        return (self.X.T.dot(self.Y - p) + lam * self.weights) / self.X.shape[0]

    def update_weights(self, new_weights):
        self.weights = new_weights

    def update_bias(self, new_bias):
        self.bias = new_bias

    def predict_labels(self, data_point):
        y_pred = np.dot(data_point, self.weights) + self.bias
        return sigmoid(y_pred)

    def train(self, learning_rate=0.005, max_epochs=500,
              lam=0.05):
        # self.method = feature_method
        # labeled_data = self.feature_extraction(labeled_data, train=True)
        ctr = 0
        prev_loss = 0
        epoch = 0
        for epoch in range(max_epochs):
            loss = 0
            self.weights -= learning_rate * self.gradient(lam)
            # Stop if loss does not change for a number of epochs
            # if abs(loss - prev_loss) < 0.00001:
            #     ctr += 1
            # prev_loss = loss
            # if ctr == 5:
            #     break
        # print(f'Training completed after epoch={epoch}.')

    def predict(self, X):
        X_r = X.reset_index(drop=True)
        w_r = self.weights.reset_index(drop=True)
        z = -np.dot(X_r, w_r)
        p = softmax(z, axis=1)
        return np.argmax(p, axis=1)
