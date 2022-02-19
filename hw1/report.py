import json
import pickle

import matplotlib.pyplot as plt


def plot_graph_logistic(plot_data):
    epochs = [25, 50, 100, 200, 500]
    lrs = [0.001, 0.005, 0.01, 0.05, 0.1]
    lams = [0.001, 0.005, 0.01, 0.05, 0.1]
    plt.style.use('seaborn-whitegrid')
    figure, axis = plt.subplots(5, 5)
    figure.tight_layout()
    # figure.subplots_adjust(hspace=0.5, wspace=0.5)
    # figure.size = (20, 20)

    title = 'Epoch: {} LR={}'

    # accuracies = [plot_data[(epochs[0], lrs[0], lam)][1] for lam in lams]
    # plt.plot(lams, accuracies)

    for i, epoch in enumerate(epochs):
        for j, lr in enumerate(lrs):
            axis[i, j].set_title(title.format(epoch, lr))
            axis[i, j].set_xlabel('Lambda')
            axis[i, j].set_ylabel('Accuracy')
            axis[i, j].set_ylim(0.75, 1.1)

            train_accuracies = [plot_data[(epoch, lr, lam)][0] for lam in lams]
            val_accuracies = [plot_data[(epoch, lr, lam)][1] for lam in lams]
            axis[i, j].plot(lams, train_accuracies, label='Training Accuracy')
            axis[i, j].plot(lams, val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.show()


def plot_graph_ensemble(plot_data):
    num_clfs = list(range(3, 11))
    plt.style.use('seaborn-whitegrid')
    # figure.subplots_adjust(hspace=0.5, wspace=0.5)
    # figure.size = (20, 20)

    plt.xlabel('Number of Classifiers')
    plt.ylabel('Accuracy')
    train_accuracies = [plot_data[num_clf][0] for num_clf in num_clfs]
    val_accuracies = [plot_data[num_clf][1] for num_clf in num_clfs]
    plt.plot(num_clfs, train_accuracies, label='Training Accuracy')
    plt.plot(num_clfs, val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.show()


def main():
    model = 'e'
    filename = 'plot_data.pkl' if model == 'l' else 'ensemble_plot_data.pkl'
    with open(filename, 'rb') as f:
        plot_data = pickle.load(f)
    if model == 'l':
        plot_graph_logistic(plot_data)
    else:
        plot_graph_ensemble(plot_data)


if __name__ == '__main__':
    main()