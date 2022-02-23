import util
from lr import Logistic
from nn3 import NeuralNetwork


data = util.prepare_data('train.csv')


def LR():
    alpha = 0.5
    lam = 0.001
    epoch = 500

    train_X, train_Y, val_X, val_Y = util.split_data(data)
    lr = Logistic(train_X, train_Y, alpha, epoch, lam)
    lr.fit()
    lr.predict(val_X, val_Y)


def NN():
    alpha = 0.01
    epoch = 200
    lam = 0.0001

    # your Multi-layer Neural Network
    train_data, train_labels, val_data, val_labels = util.split_data(data)

    neural_net = NeuralNetwork(train_data, train_labels, alpha=alpha, lam=lam, epochs=epoch, hidden_neurons=37)
    neural_net.fit()
    neural_net.predict(val_data, val_labels)


if __name__ == '__main__':
    # test_X, test_Y = util.prepare_data('test.csv')
    print("..................Beginning of Logistic Regression................")
    LR()
    print("..................End of Logistic Regression................")

    print("------------------------------------------------")

    print("..................Beginning of Neural Network................")
    # NN()
    print("..................End of Neural Network................")
