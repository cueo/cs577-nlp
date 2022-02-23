import numpy as np
from sklearn.metrics import f1_score


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

    def predict(self, X, Y):
        z = -np.dot(X, self.weights)
        p = softmax(z)
        predictions = np.argmax(p, axis=1)
        labels = Y
        if not len(Y.shape) == 1:
            labels = np.argmax(Y, axis=1)
        print(f'{np.unique(predictions)} / {np.unique(labels)}')
        accuracy = np.mean(predictions == labels)
        f1 = f1_score(labels, predictions, average='macro')
        print(f'Accuracy: {accuracy} F1: {f1}')
        return accuracy, f1
