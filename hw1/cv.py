import os.path
import pickle

import numpy as np

import util
from lr import Logistic
from nn3 import NeuralNetwork


def cross_validation_lr(data, epochs, alphas, lams):
    plot_data_path = 'models/lr_plot_data_2.pkl'

    plot_data = {}
    if os.path.exists(plot_data_path):
        with open(plot_data_path, 'wb') as f:
            plot_data = pickle.load(f)
    folds = 5
    for epoch in epochs:
        for alpha in alphas:
            for lam in lams:
                if (epoch, alpha, lam) in plot_data:
                    print(f'Epoch: {epoch}, alpha: {alpha} already in plot_data')
                    continue

                train_accuracies = []
                test_accuracies = []
                train_f1s = []
                test_f1s = []
                for i in range(folds):
                    train_data, train_labels, test_data, test_labels = util.cv_split_data(data, i)
                    model = Logistic(train_data, train_labels, learning_rate=alpha, max_epochs=epoch, lam=lam)
                    model.fit()

                    train_accuracy, train_f1 = model.predict(train_data, train_labels)
                    test_accuracy, test_f1 = model.predict(test_data, test_labels)
                    train_accuracies.append(train_accuracy)
                    train_f1s.append(train_f1)
                    test_accuracies.append(test_accuracy)
                    test_f1s.append(test_f1)

                    print(f'Epoch: {epoch}, alpha: {alpha}, lam: {lam}, fold: {i},train_accuracy: {train_accuracy},'
                          f'test_accuracy: {test_accuracy}')

                train_accuracy = np.mean(train_accuracies)
                test_accuracy = np.mean(test_accuracies)
                train_f1 = np.mean(train_f1s)
                test_f1 = np.mean(test_f1s)
                plot_data[(epoch, alpha, lam)] = {
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_f1': train_f1,
                    'test_f1': test_f1
                }
                with open(plot_data_path, 'wb') as f:
                    pickle.dump(plot_data, f)
    return plot_data


def cross_validation_nn(data, epochs, alphas, lams, hidden_neurons):
    plot_data_path = 'models/nn_plot_data_2.pkl'

    plot_data = {}
    folds = 5
    for epoch in epochs:
        for alpha in alphas:
            for lam in lams:
                for hidden_neuron in hidden_neurons:
                    # if os.path.exists(plot_data_path):
                    #     with open(plot_data_path, 'wb') as f:
                    #         plot_data = pickle.load(f)
                    # if (epoch, alpha, lam, hidden_neuron) in plot_data:
                    #     print(f'Epoch: {epoch}, alpha: {alpha} already in plot_data')
                    #     continue
                    train_accuracies = []
                    test_accuracies = []
                    train_f1s = []
                    test_f1s = []
                    for i in range(folds):
                        train_data, train_labels, test_data, test_labels = util.cv_split_data(data, i)
                        model = NeuralNetwork(train_data, train_labels, alpha=alpha, lam=lam, epochs=epoch,
                                              hidden_neurons=hidden_neuron)
                        model.fit()

                        train_accuracy, train_f1 = model.predict(train_data, train_labels)
                        test_accuracy, test_f1 = model.predict(test_data, test_labels)
                        train_accuracies.append(train_accuracy)
                        train_f1s.append(train_f1)
                        test_accuracies.append(test_accuracy)
                        test_f1s.append(test_f1)

                        print(
                            f'Epoch: {epoch}, alpha: {alpha}, lam: {lam}, hidden_neuron: {hidden_neuron}, fold: {i}, '
                            f'train_accuracy: {train_accuracy}, test_accuracy: {test_accuracy}')

                    train_accuracy = np.mean(train_accuracies)
                    test_accuracy = np.mean(test_accuracies)
                    train_f1 = np.mean(train_f1s)
                    test_f1 = np.mean(test_f1s)
                    plot_data[(epoch, alpha, lam, hidden_neuron)] = {
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'train_f1': train_f1,
                        'test_f1': test_f1
                    }
                    with open(plot_data_path, 'wb') as f:
                        pickle.dump(plot_data, f)
    return plot_data


def lr():
    data = util.prepare_data('train.csv')
    # epochs = [100, 200, 500, 1000]
    lrs = [0.001, 0.01, 0.1, 1]
    # lams = [0.001, 0.0001]
    epochs = [1000]
    lams = [0.0001]
    cross_validation_lr(data, epochs=epochs, alphas=lrs, lams=lams)


def nn():
    data = util.prepare_data('train.csv')
    # epochs = [100, 200, 250, 300]
    alphas = [0.001, 0.01, 0.05, 0.1]
    # lams = [0.001, 0.0001]
    # hidden_neurons = [50, 100]
    epochs = [300]
    lams = [0.0001]
    hidden_neurons = [100]

    cross_validation_nn(data, epochs=epochs, alphas=alphas, lams=lams, hidden_neurons=hidden_neurons)


def main():
    lr()
    # nn()


if __name__ == '__main__':
    main()
