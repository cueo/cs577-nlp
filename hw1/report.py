import pickle

import matplotlib.pyplot as plt


def plot_graph(plot_data, model):
    alphas = {k[1] for k in plot_data.keys()}
    x = sorted(alphas)

    if model == 'lr':
        title = 'Logistic Regression'
        epoch, alpha, lam = 0, 0, 0
        params = (epoch, alpha, lam)
    else:
        title = 'Neural Network'
        epoch, alpha, lam, hn = 0, 0, 0, 0
        params = (epoch, alpha, lam, hn)
    test_accuracy = 0
    for k, v in plot_data.items():
        if v['test_accuracy'] > test_accuracy:
            params = k

    print(f'Parameters: {params}')

    train_accuracies = []
    test_accuracies = []
    train_f1s = []
    test_f1s = []
    for alpha in x:
        if model == 'lr':
            key = (params[0], alpha, params[2])
        else:
            key = (params[0], alpha, params[2], params[3])
        train_accuracies.append(plot_data[key]['train_accuracy'])
        test_accuracies.append(plot_data[key]['test_accuracy'])
        train_f1s.append(plot_data[key]['train_f1'])
        test_f1s.append(plot_data[key]['test_f1'])

    # draw(x, train_accuracies, test_accuracies, title, 'Accuracy')
    draw(x, train_f1s, test_f1s, title, 'F1 Score')


def draw(x, y1, y2, title, label):
    fig, ax = plt.subplots()
    ax.plot(x, y1, '-o', label=f'Train {label}')
    ax.plot(x, y2, '-o', label=f'Validation {label}')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel(f'{label}')
    ax.set_title(title)
    ax.legend()
    plt.show()


def main():
    # model = 'lr'
    model = 'nn'
    filename = 'lr_plot_data_2.pkl' if model == 'lr' else 'nn_plot_data_2.pkl'
    with open(f'models/{filename}', 'rb') as f:
        plot_data = pickle.load(f)
    plot_graph(plot_data, model)


if __name__ == '__main__':
    main()
