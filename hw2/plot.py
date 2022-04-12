import json

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from dmemm import train, train_gru, predict
from embeddings import word_embeddings, tag_embeddings
from starter2 import *


# def find_highest_score(results):
#     max_key = ''
#     max_value = 0
#     for hyperparams, scores in results.items():
#         if scores['test']['f1'] > max_value:
#             max_key = hyperparams
#             max_value = scores['test']['f1']
#     return max_key, max_value
#
#
# def run(epochs, hidden_dim, lr, option,
#         word_embeds, word_to_idx, word_to_embeddings, tag_to_embeddings):
#     train_fn = train
#     if option == 3:
#         train_fn = train_gru
#     train_X, valid_X, train_Y, valid_Y = train_test_split(train_sents, train_tags, test_size=0.2, random_state=42)
#     model = train_fn(train_X, train_Y, word_embeds, word_to_embeddings, tag_to_embeddings,
#                      hidden_dim=hidden_dim, lr=lr, epochs=epochs)
#     train_precision, train_recall, train_f1 = predict(train_X, train_Y, model, word_embeds, word_to_idx,
#                                                       tag_to_embeddings)
#     valid_precision, valid_recall, valid_f1 = predict(valid_X, valid_Y, model, word_embeds, word_to_idx,
#                                                       tag_to_embeddings)
#     test_precision, test_recall, test_f1 = predict(test_sents, test_tags, model, word_embeds, word_to_idx,
#                                                    tag_to_embeddings)
#     results = {
#         'train': {
#             'precision': train_precision,
#             'recall': train_recall,
#             'f1': train_f1
#         },
#         'validation': {
#             'precision': valid_precision,
#             'recall': valid_recall,
#             'f1': valid_f1
#         },
#         'test': {
#             'precision': test_precision,
#             'recall': test_recall,
#             'f1': test_f1
#         }
#     }
#     return results
#
#
# def run_best_model(results, hyperparameter):
#     best_model, best_score = find_highest_score(results)
#     print(f'Best model: {best_model} with score: {best_score}')
#     # 1_100_64_0.001
#     option, epochs, hidden_dim, lr = map(float, best_model.split('_'))
#     option, epochs, hidden_dim = map(int, (option, epochs, hidden_dim))
#
#     train_f1s = []
#     valid_f1s = []
#     test_f1s = []
#
#     word_embeds, word_to_idx, word_to_embeddings = word_embeddings(option=option)
#     _, tag_to_embeddings = tag_embeddings()
#
#     if hyperparameter == 'lr':
#         x = [0.0001, 0.005, 0.001, 0.05, 0.01, 0.1]
#         for lr in x:
#             print(f'Running model with lr: {lr}')
#             results = run(epochs, hidden_dim, lr, option,
#                           word_embeds, word_to_idx, word_to_embeddings, tag_to_embeddings)
#             train_f1s.append(results['train']['f1'])
#             valid_f1s.append(results['validation']['f1'])
#             test_f1s.append(results['test']['f1'])
#     else:
#         x = [50, 100, 200, 300, 400, 500]
#         for epochs in x:
#             print(f'Running model with epochs: {epochs}')
#             results = run(epochs, hidden_dim, lr, option,
#                           word_embeds, word_to_idx, word_to_embeddings, tag_to_embeddings)
#             train_f1s.append(results['train']['f1'])
#             valid_f1s.append(results['validation']['f1'])
#             test_f1s.append(results['test']['f1'])
#     return x, train_f1s, valid_f1s, test_f1s


def plot(x, train_f1s, valid_f1s, test_f1s, hyperparameter):
    fig, ax = plt.subplots()
    ax.plot(x, train_f1s, '-o', label=f'Train F1 Score')
    ax.plot(x, valid_f1s, '-o', label=f'Validation F1 Score')
    ax.plot(x, test_f1s, '-o', label=f'Test F1 Score')
    ax.set_xlabel(hyperparameter)
    ax.set_ylabel('F1 Score')
    ax.set_title('Deep Maximum Entropy Markov Model')
    ax.legend()
    plt.show()


def main():
    # with open('plot_results.json', 'r') as f:
    #     results = json.load(f)
    # hyperparameters = {'lr', 'epochs'}
    # for hyperparameter in hyperparameters:
    #     x, train_f1s, valid_f1s, test_f1s = run_best_model(results, hyperparameter)
    #     print(f'Hyperparameter: {hyperparameter} = {x}')
    #     print(f'Train F1 Score: {train_f1s}')
    #     print(f'Validation F1 Score: {valid_f1s}')
    #     print(f'Test F1 Score: {test_f1s}')
    #     print()
    # x = [50, 100, 200, 300, 400, 500]
    # train_f1s = [0, 0.33213429256594723, 0.2671996633705028, 0.3869241062894283, 0.38921001926782267, 0.37037037037037035]
    # valid_f1s = [0, 0.231433506044905, 0.27709861450692747, 0.2859399684044234, 0.30126182965299686, 0.309748427672956]
    # test_f1s = [0, 0.2893258426966292, 0.35685752330226367, 0.36781609195402293, 0.41405269761606023, 0.40806045340050384]
    # plot(x[1:], train_f1s[1:], valid_f1s[1:], test_f1s[1:], 'Epochs')

    # x = [50, 100, 200, 300, 400, 500]
    # train_f1s = [0.3043657331136738, 0.4020777222008465, 0.562968248305387, 0.5905898876404494, 0.5836575875486382, 0.5941495883692416]
    # valid_f1s = [0.25610783487784333, 0.28865979381443296, 0.3041825095057034, 0.3226765799256506, 0.3148425787106447, 0.32547864506627394]
    # test_f1s = [0.34206896551724136, 0.3585147247119078, 0.36904761904761907, 0.36923076923076925, 0.37850467289719636, 0.37132784958871906]
    # plot(x, train_f1s, valid_f1s, test_f1s, 'Epochs')

    # x = [50, 100, 200, 300, 400, 500]
    # train_f1s = [0.3043657331136738, 0.4020777222008465, 0.562968248305387, 0.5905898876404494, 0.5836575875486382, 0.5941495883692416]
    # valid_f1s = [0.25610783487784333, 0.28865979381443296, 0.3041825095057034, 0.3226765799256506, 0.3148425787106447, 0.32547864506627394]
    # test_f1s = [0.34206896551724136, 0.3585147247119078, 0.36904761904761907, 0.36923076923076925, 0.37850467289719636, 0.37132784958871906]
    # plot(x, train_f1s, valid_f1s, test_f1s, 'Epochs')

    x = [0.001, 0.005, 0.01, 0.05, 0.1]
    train_f1s = [0.1760532889258951, 0.6825573624415238, 0.6127352823388066, 0.3420237010027347, 0.14243146603098927]
    valid_f1s = [0.15286789514663898, 0.2895705521472393, 0.26138123257974605, 0.15425181278839814, 0.07359009628610728]
    test_f1s = [0.17922705314009663, 0.3217821782178218, 0.2918454935622318, 0.20025673940949937, 0.06179775280898876]
    plot(x, train_f1s, valid_f1s, test_f1s, 'Learning Rate')


if __name__ == '__main__':
    main()
