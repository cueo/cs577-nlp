import json
import os.path
import pickle

from dmemm import predict, train, train_gru
from embeddings import word_embeddings, tag_embeddings
from starter2 import train_sents, train_tags, test_sents, test_tags


def save_results(epochs, f1, hidden_dim, lr, option, precision, recall):
    with open('results.json', 'r') as f:
        results = json.load(f)
    with open('results.json', 'w') as f:
        key = f'{option}_{epochs}_{hidden_dim}_{lr}'
        if key not in results or results[key]['f1'] < f1:
            results[key] = {
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        json.dump(results, f, indent=2)


def run(option, hidden_dim, lr, epochs):
    # option = 1
    # hidden_dim = 128
    # lr = 0.01
    # lam = 0.01
    # epochs = 100

    filepath = f'models/model_{option}_{epochs}_{hidden_dim}_{lr}.pkl'
    word_embeds, word_to_idx, word_to_embeddings = word_embeddings(option=option)
    _, tag_to_embeddings = tag_embeddings()

    train_fn = train
    if option == 3:
        train_fn = train_gru

    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    else:
        model = train_fn(train_sents, train_tags, word_embeds, word_to_embeddings, tag_to_embeddings,
                         hidden_dim=hidden_dim, lr=lr, epochs=epochs)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    predict(train_sents, train_tags, model, word_embeds, word_to_idx, tag_to_embeddings)
    precision, recall, f1 = predict(test_sents, test_tags, model, word_embeds, word_to_idx, tag_to_embeddings)
    save_results(epochs, f1, hidden_dim, lr, option, precision, recall)


def run_all():
    with open('results.json', 'r') as f:
        models = json.load(f)

    options = [1, 2]
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


def run_one(option=1, hidden_dim=64, lr=0.05, epochs=50):
    run(option=option, hidden_dim=hidden_dim, lr=lr, epochs=epochs)


def run_gru():
    option = 2
    hidden_dim = 128
    lr = 0.05
    epochs = 50
    filepath = f'models/model_gru_{option}_{epochs}_{hidden_dim}_{lr}.pkl'
    word_embeds, word_to_idx, word_to_embeddings = word_embeddings(option=option)
    _, tag_to_embeddings = tag_embeddings()

    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    else:
        model = train_gru(train_sents, train_tags, word_embeds, word_to_embeddings, tag_to_embeddings,
                          hidden_dim=hidden_dim, lr=lr, epochs=epochs)
        if model is not None:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
    predict(train_sents, train_tags, model, word_embeds, word_to_idx, tag_to_embeddings)
    precision, recall, f1 = predict(test_sents, test_tags, model, word_embeds, word_to_idx, tag_to_embeddings)
    save_results(epochs, f1, hidden_dim, lr, option, precision, recall)


def main():
    # run_one()
    run_gru()


if __name__ == '__main__':
    main()
