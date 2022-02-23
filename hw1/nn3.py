import numpy as np
from sklearn.metrics import f1_score


def cross_entropy(y, y_pred):
    y_pred += 1e-7
    return -np.sum(y * np.log(y_pred)) / y.shape[0]


def mse(y, y_pred):
    return np.mean(np.square(y - y_pred))


def loss_fn(y, y_pred):
    return cross_entropy(y, y_pred)
    # return mse(y, y_pred)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(X):
    # e_x = np.exp(x - np.max(x))
    # return e_x / e_x.sum()

    ps = np.empty(X.shape)
    for i in range(X.shape[0]):
        ps[i, :] = np.exp(X[i, :])
        ps[i, :] /= np.sum(ps[i, :])
    return ps

    # max_value = np.max(z, axis=1)
    # max_value = max_value.reshape(max_value.shape[0], 1)
    # exponent = np.exp(z - max_value)
    # summation = exponent.sum(axis=1, keepdims=True)
    # # summation = np.sum(exponent,axis=1)
    # summation = summation.reshape(summation.shape[0], 1)
    # # print(summation)
    # return exponent / summation


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

    def predict(self, inputs, labels):
        # Forward pass
        z1 = np.dot(inputs, self.w1) + self.b1
        a1 = sigmoid(z1)

        z2 = np.dot(a1, self.w2) + self.b2
        a2 = softmax(z2)
        predictions = np.argmax(a2, axis=1)
        labels = np.argmax(labels, axis=1)
        # print(f'{np.unique(predictions)} / {np.unique(labels)}')
        accuracy = np.mean(predictions == labels)
        f1 = f1_score(predictions, labels, average='macro')
        print(f'Accuracy: {accuracy} F1: {f1}')
        return accuracy, f1
