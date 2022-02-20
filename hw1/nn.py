def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2


# noinspection PyPep8Naming
class Layer:
    def __init__(self, n_inputs, n_outputs, activation_function, activation_function_derivative):
        self.weights = 0.10 * np.random.randn(n_inputs, n_outputs)
        self.bias = np.zeros((1, n_outputs))

        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

        self.X = None
        self.Y = None

    def forward_propagate(self, data):
        self.X = data
        z = np.dot(X, self.weights) + self.bias
        self.Y = self.activation_function(self.z)
        return self.Y

    def backward_propagate(self, dL_dY, learning_rate):
        dL_dW = np.dot(self.X.T, dL_dY) * self.activation_function_derivative(self.Y)
        self.weights -= learning_rate * dL_dW


class NeuralNetwork:
    def __init__(self, train, labels):
        self.train = train
        self.labels = labels

        self.layers = []
        self.add_layers()

    def add_layers(self):
        self.layers.append(Layer(self.train.shape[1], 500, tanh, tanh_derivative))
        self.layers.append(Layer(500, self.labels.shape[1], tanh, tanh_derivative))

    def fit(self, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            for i in range(len(self.train)):
                x = self.train[i]
                y = self.labels[i]
                output = x
                for layer in self.layers:
                    output = layer.forward_propagate(output)
                error = mse_derivative(y, x)
                self.backward_propagate(y, learning_rate)

    def predict(self, x):
        return self.forward_propagate(x)
