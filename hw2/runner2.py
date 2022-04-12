import json
import os.path
import pickle

from sklearn.model_selection import train_test_split

from dmemm import predict, train, train_gru
from embeddings import word_embeddings, tag_embeddings
from starter import train_sents, train_tags, test_sents, test_tags


def save_results(option, epochs, hidden_dim, lr, result):
    with open('plot_results.json', 'r') as f:
        results = json.load(f)
    with open('plot_results.json', 'w') as f:
        key = f'{option}_{epochs}_{hidden_dim}_{lr}'
        if key not in results or results[key]['test']['f1'] < result['test']['f1']:
            results[key] = result
        json.dump(results, f, indent=2)


def run(option, hidden_dim, lr, epochs):
    # option = 1
    # hidden_dim = 128
    # lr = 0.01
    # lam = 0.01
    # epochs = 100

    word_embeds, word_to_idx, word_to_embeddings = word_embeddings(option=option)
    _, tag_to_embeddings = tag_embeddings()

    train_fn = train
    if option == 3:
        train_fn = train_gru

    train_X, valid_X, train_Y, valid_Y = train_test_split(train_sents, train_tags, test_size=0.2, random_state=42)

    model = train_fn(train_X, train_Y, word_embeds, word_to_embeddings, tag_to_embeddings,
                     hidden_dim=hidden_dim, lr=lr, epochs=epochs)
    train_precision, train_recall, train_f1 = predict(train_X, train_Y, model, word_embeds, word_to_idx,
                                                      tag_to_embeddings)
    valid_precision, valid_recall, valid_f1 = predict(valid_X, valid_Y, model, word_embeds, word_to_idx,
                                                      tag_to_embeddings)
    test_precision, test_recall, test_f1 = predict(test_sents, test_tags, model, word_embeds, word_to_idx,
                                                   tag_to_embeddings)
    results = {
        'train': {
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1
        },
        'validation': {
            'precision': valid_precision,
            'recall': valid_recall,
            'f1': valid_f1
        },
        'test': {
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1
        }
    }
    save_results(option, epochs, hidden_dim, lr, results)


def run_all():
    with open('plot_results.json', 'r') as f:
        models = json.load(f)

    options = [3, 1, 2]
    hidden_dims = [64, 128]
    lrs = [0.01, 0.001]
    epochs = [50, 100, 200]
    for option in options:
        for hidden_dim in hidden_dims:
            for lr in lrs:
                for epoch in epochs:
                    if f'{option}_{epoch}_{hidden_dim}_{lr}' in models:
                        continue
                    print(f'Training with hyperparameters: '
                          f'option={option}, hidden_dim={hidden_dim}, lr={lr}, epochs={epoch}')
                    run(option=option, hidden_dim=hidden_dim, lr=lr, epochs=epoch)
                    print('\n')


def run_one():
    option = 1
    hidden_dim = 64
    lr = 0.05
    epochs = 50
    run(option=option, hidden_dim=hidden_dim, lr=lr, epochs=epochs)


def main():
    # run_one()
    # run_gru()
    run_all()


if __name__ == '__main__':
    main()
