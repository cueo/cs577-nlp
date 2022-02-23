import numpy as np


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


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class NeuralNetwork:
    def __init__(self, train, labels, alpha, lam, epochs):
        self.train = train
        self.labels = labels
        self.alpha = alpha
        self.lam = lam
        self.epochs = epochs

        self.weights = []
        self.bias = []
        self.layers = []
        self.outputs = []
        self.activations = []
        self.activations_deriv = []
        self.hidden_layers = 1
        self.add_layers(train.shape[1], labels.shape[1], self.hidden_layers)

    def add_layers(self, input_neurons, output_neurons, hidden_layers, activation=sigmoid,
                   activation_deriv=sigmoid_derivative):
        hidden_neurons = int(input_neurons // 2)
        for layer in range(hidden_layers):
            self.weights.append(np.random.randn(input_neurons, hidden_neurons) * 0.1)
            self.bias.append(np.random.randn(1, hidden_neurons) * 0.1)
            self.activations.append(activation)
            self.activations_deriv.append(activation_deriv)
            self.outputs.append(np.zeros((1, hidden_neurons)))
            input_neurons = hidden_neurons
            hidden_neurons = int(input_neurons // 2)
        self.weights.append(np.random.randn(input_neurons, output_neurons) * 0.1)
        self.bias.append(np.random.randn(1, output_neurons) * 0.1)
        self.activations.append(softmax)
        self.activations_deriv.append(None)
        self.outputs.append(np.zeros((1, output_neurons)))

    def feed_forward(self, inputs):
        for layer in range(self.hidden_layers):
            z = np.dot(inputs, self.weights[layer]) + self.bias[layer]
            self.outputs[layer] = self.activations[layer](z)

        # Output layer
        z = np.dot(self.outputs[-2], self.weights[-1]) + self.bias[-1]
        self.outputs[-1] = self.activations[-1](z)
        return self.outputs[-1]

    def backprop(self, x, y):
        # Output layer
        error = self.outputs[-1] - y
        weights_delta = np.dot(self.outputs[-2].T, error)  # outputs[-2] is the output of the second last layer
        self.weights[-1] -= (self.alpha * weights_delta) + self.lam * self.weights[-1]
        self.bias[-1] -= self.alpha * error

        # Hidden layers - shouldn't go inside if there is only one hidden layer
        for layer in range(self.hidden_layers-1, 1, -1):
            # error = np.dot(error, self.weights[layer].T) * self.activations_deriv[layer](self.outputs[layer - 1])
            # self.weights[layer] -= self.alpha * np.dot(self.outputs[layer - 1].T, error)
            # self.bias[layer] -= self.alpha * error
            error = np.dot(self.weights[layer+1], error.T)
            dW = np.multiply(error, self.activations_deriv[layer](self.outputs[layer]))
            weights_delta = np.dot(x.T, dW)

        # First hidden layer
        error = np.dot(self.weights[1], error.T)
        dW = np.multiply(error.T, self.activations_deriv[0](self.outputs[0]))
        weights_delta = np.dot(x.T, dW)
        self.weights[0] -= (self.alpha * weights_delta) + self.lam * self.weights[0]
        self.bias[0] -= self.alpha * dW

    def fit(self):
        for epoch in range(self.epochs):
            loss = 0
            for i in range(self.train.shape[0]):
                x = self.train[i].reshape(1, self.train.shape[1])
                output = self.feed_forward(x)
                self.backprop(x, self.labels[i])

                loss += loss_fn(self.labels[i], output)

            print("Epoch: {}, Loss: {}".format(epoch, loss))

    def predict(self, inputs, labels):
        predictions = []
        for x in inputs:
            output = self.feed_forward(x)
            predictions.append(output)
        print(f'Accuracy: {np.mean(predictions == labels)}')
