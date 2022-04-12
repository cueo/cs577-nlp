import json

from sklearn.model_selection import train_test_split

from dmemm import train, train_gru, predict
from embeddings import word_embeddings, tag_embeddings
from starter import *


def find_highest_score_for_model(results, option):
    max_key = ''
    max_value = 0
    for hyperparams, scores in results.items():
        if hyperparams.startswith(str(option)):
            if scores['test']['f1'] > max_value:
                max_key = hyperparams
                max_value = scores['test']['f1']
    return max_key, max_value


def run(epochs, hidden_dim, lr, option,
        word_embeds, word_to_idx, word_to_embeddings, tag_to_embeddings):
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
    return results


def run_best_model(plot_results):
    options = [1, 2, 3]
    x = [50, 100, 200, 300, 400, 500]
    train_f1s = {}
    valid_f1s = {}
    test_f1s = {}
    details = {}

    for option in options:
        best_model, best_score = find_highest_score_for_model(plot_results, option)
        print(f'Best model: {best_model} with score: {best_score} for option: {option}')
        # 1_100_64_0.001
        option, epochs, hidden_dim, lr = map(float, best_model.split('_'))
        option, epochs, hidden_dim = map(int, (option, epochs, hidden_dim))

        details[option] = {
            'epochs': epochs,
            'hidden_dim': hidden_dim,
            'lr': lr
        }

        train_f1 = []
        valid_f1 = []
        test_f1 = []

        word_embeds, word_to_idx, word_to_embeddings = word_embeddings(option=option)
        _, tag_to_embeddings = tag_embeddings()

        for epochs in x:
            print(f'Running model with epochs: {epochs}')
            results = run(epochs, hidden_dim, lr, option,
                          word_embeds, word_to_idx, word_to_embeddings, tag_to_embeddings)
            train_f1.append(results['train']['f1'])
            valid_f1.append(results['validation']['f1'])
            test_f1.append(results['test']['f1'])
        train_f1s[option] = train_f1
        valid_f1s[option] = valid_f1
        test_f1s[option] = test_f1
    return x, train_f1s, valid_f1s, test_f1s, details


def main():
    with open('plot_results.json', 'r') as f:
        results = json.load(f)
    x, train_f1s, valid_f1s, test_f1s, details = run_best_model(results)
    print(f'Epochs: {x}')
    print(f'Train F1 Score: {train_f1s}')
    print(f'Validation F1 Score: {valid_f1s}')
    print(f'Test F1 Score: {test_f1s}')
    print(f'Hyperparameters: {details}')


if __name__ == '__main__':
    main()
