from typing import List

import numpy as np

LOG_TIME = False


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


def cross_entropy(y, y_pred):
    y_pred += 1e-7
    return -np.sum(y * np.log(y_pred)) / y.shape[0]


def cross_entropy_derivative(y, y_pred):
    return -y / y_pred


def mse(y, y_pred):
    return np.mean(np.square(y - y_pred))


def loss_fn(y, y_pred):
    return cross_entropy(y, y_pred)
    # return mse(y, y_pred)


# noinspection PyPep8Naming
def softmax(X):
    # e_x = np.exp(x - np.max(x))
    # return e_x / e_x.sum()
    ps = np.empty(X.shape)
    for i in range(X.shape[0]):
        x = X[i, :]
        ps[i, :] = np.exp(x - np.max(x))
        ps[i, :] /= np.sum(ps[i, :])
    return ps


def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))


# noinspection PyPep8Naming
class Layer:
    def __init__(self, n_inputs, n_outputs, activation_function, activation_function_derivative, output=False):
        self.weights = 0.10 * np.random.randn(n_inputs, n_outputs)
        self.bias = np.zeros((1, n_outputs))

        self.activation = activation_function
        self.activation_deriv = activation_function_derivative

        self.X = None
        self.Y = None

        self.output = output

    def feedforward(self, data):
        self.X = data
        z = np.dot(self.X, self.weights) + self.bias
        self.Y = self.activation(z)
        return self.Y

    def backprop(self, dL_dY, W_prev, learning_rate, lam):
        dW = dL_dY
        if not self.output:
            dW = np.dot(dL_dY, W_prev.T) * self.activation_deriv(self.Y)
        dB = np.sum(dW, axis=0)
        self.weights -= learning_rate * np.dot(self.X.T, dW) + lam * self.weights
        self.bias -= learning_rate * dB + lam * self.bias
        return dW


def to_normalized_np_array(train):
    # train = train.apply(lambda x: x / x.sum())
    # return train.to_numpy().reshape(train.shape[0], 1, -1)
    return train.to_numpy()


# noinspection PyPep8Naming
class NeuralNetwork:
    def __init__(self, train, labels):
        # self.train = to_normalized_np_array(train)[:2]
        # self.labels = util.encode_onehot(labels)[:2]
        self.train = train
        self.labels = labels
        self.output_size = self.labels.shape[1]

        self.layers: List[Layer] = []
        self.add_layers()

    def add_layers(self):
        self.layers.append(Layer(self.train.shape[1], 100, sigmoid, sigmoid_derivative))
        self.layers.append(Layer(100, self.output_size, softmax, None, output=True))

    def fit(self, learning_rate=0.01, epochs=500, lam=0.001):
        batch_size = self.train.shape[0]
        for epoch in range(epochs):
            loss = 0
            for i in range(0, len(self.train), batch_size):
                x = self.train[i: i + batch_size]
                y = self.labels[i: i + batch_size]
                output = x
                for layer in self.layers:
                    output = layer.feedforward(output)

                loss += loss_fn(y, output[0])

                error = output - y
                w_prev = None
                for layer in reversed(self.layers):
                    error = layer.backprop(error, w_prev, learning_rate, lam)
                    w_prev = layer.weights
            print(f'Epoch: {epoch}, Loss: {loss / len(self.train)}')

    def predict(self, X):
        predictions = []
        for x in X:
            output = x
            for layer in self.layers:
                output = layer.feedforward(output)
            predictions.append(np.argmax(output))
        return predictions
