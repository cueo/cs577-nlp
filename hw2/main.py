import argparse

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

from dmemm import train, train_gru, predict
from embeddings import word_embeddings, tag_embeddings

print('Loading word embeddings...')
# wv_from_bin = KeyedVectors.load_word2vec_format(datapath('/homes/cs577/hw2/w2v.bin'), binary=True)
wv_from_bin = KeyedVectors.load_word2vec_format(datapath('/Users/cueball/dev/purdue/cs577/hw2/data/w2v.bin'),
                                                binary=True)
print('Loaded word embeddings.')


def load_data(path, lowercase=True):
    """
    Load training and testing data.
    """
    sents = []
    tags = []
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            sent = []
            tag = []
            for pair in line.split('####')[1].split(' '):
                tn, tg = pair.rsplit('=', 1)
                if lowercase:
                    sent.append(tn.lower())
                else:
                    sent.append(tn)
                tag.append(tg)
            sents.append(sent)
            tags.append(tag)
    return sents, tags


def run(train_sents, train_tags, test_sents, test_tags,
        option, hidden_dim, lr, epochs, test_path):
    # option = 1
    # hidden_dim = 128
    # lr = 0.01
    # lam = 0.01
    # epochs = 100

    word_embeds, word_to_idx, word_to_embeddings = word_embeddings(option, train_sents, wv_from_bin)
    _, tag_to_embeddings = tag_embeddings()

    train_fn = train
    if option == 3:
        train_fn = train_gru

    model = train_fn(train_sents, train_tags, word_embeds, word_to_embeddings, tag_to_embeddings,
                     hidden_dim=hidden_dim, lr=lr, epochs=epochs)
    # predict(option, train_sents, train_tags, model, word_embeds, word_to_idx, tag_to_embeddings)
    predict(option, test_sents, test_tags, model, word_embeds, word_to_idx, tag_to_embeddings, test_path=test_path)


def get_args():
    parser = argparse.ArgumentParser(description='Deep Maximum Entropy Markov Model')
    parser.add_argument('--train_file', type=str, default='data/twitter1_train.txt', help='training file path')
    parser.add_argument('--test_file', type=str, default='data/twitter1_test.txt', help='test file path')
    parser.add_argument('--option', type=int, required=True, help='embedding option')
    args = parser.parse_args()
    train_file = args.train_file
    test_file = args.test_file
    option = args.option
    return option, train_file, test_file


def main():
    option, train_file, test_file = get_args()

    train_sents, train_tags = load_data(train_file)
    test_sents, test_tags = load_data(test_file)

    if option == 1:
        epochs = 100
        hidden_dim = 128
        lr = 0.01
    elif option == 2:
        epochs = 100
        hidden_dim = 128
        lr = 0.05
    else:
        epochs = 100
        hidden_dim = 128
        lr = 0.01

    run(train_sents, train_tags, test_sents, test_tags,
        option, hidden_dim, lr, epochs, test_file)


if __name__ == '__main__':
    main()
