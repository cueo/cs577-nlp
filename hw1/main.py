import pickle

import numpy as np

import util
from const import reverse_labels_dict
from lr import Logistic


to_label = np.vectorize(lambda x: reverse_labels_dict[x])


def cross_validation(data, labels, classifier):
    plot_data = {}
    folds = 5
    epochs = [100, 200, 500, 1000]
    lrs = [0.001, 0.005, 0.01, 0.1]
    lams = [0.001, 0.005, 0.01, 0.1]
    for epoch in epochs:
        for lr in lrs:
            for lam in lams:
                train_accuracies = []
                test_accuracies = []
                for i in range(folds):
                    train_data, train_labels, test_data, test_labels = util.cv_split_data(data, labels, i)
                    model = classifier(train_data, train_labels)
                    model.fit(learning_rate=lr, max_epochs=epoch, lam=lam)

                    predicted_train_labels = model.predict(train_data)
                    predicted_test_labels = model.predict(test_data)

                    train_accuracy = util.accuracy(train_labels, predicted_train_labels)
                    train_accuracies.append(train_accuracy)
                    test_accuracy = util.accuracy(test_labels, predicted_test_labels)
                    test_accuracies.append(test_accuracy)

                    print(f'Epoch: {epoch}, lr: {lr}, lam: {lam}, fold: {i},train_accuracy: {train_accuracy},'
                          f'test_accuracy: {test_accuracy}')

                train_accuracy = np.mean(train_accuracies)
                test_accuracy = np.mean(test_accuracies)
                plot_data[(epoch, lr, lam)] = (train_accuracy, test_accuracy)
    return plot_data


def LR():
    # your logistic regression
    data, labels = util.prepare_data('train.csv')
    plot_data = cross_validation(data, labels, Logistic)
    with open('models/lr_plot_data.pkl', 'wb') as f:
        pickle.dump(plot_data, f)

    # train_X, train_Y, val_X, val_Y = util.split_data(X, Y)
    # lr = Logistic(train_X, train_Y)
    # lr.train()
    # predictions = lr.predict(val_X)
    # print(f'Emotions detected: {to_label(np.unique(predictions))}, emotions in test set: {to_label(np.unique(val_Y))}')
    # accuracy = util.accuracy(predictions, val_Y)
    # print(f'Accuracy: {accuracy}')


def NN():
    pass
    # your Multi-layer Neural Network


if __name__ == '__main__':
    # test_X, test_Y = util.prepare_data('test.csv')
    print("..................Beginning of Logistic Regression................")
    LR()
    print("..................End of Logistic Regression................")

    print("------------------------------------------------")

    print("..................Beginning of Neural Network................")
    NN()
    print("..................End of Neural Network................")
