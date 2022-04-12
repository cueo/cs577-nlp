# Load training and testing data
import os.path


def load_data(path, lowercase=True):
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


train_sents, train_tags = load_data('data/twitter1_train.txt')
test_sents, test_tags = load_data('data/twitter1_test.txt')

# Load Word2Vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

wv_model_path = 'data/kv.model'
if not os.path.exists(wv_model_path):
    wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/Users/cueball/dev/purdue/cs577/hw2/data/w2v.bin"),
                                                    binary=True)
    wv_from_bin.init_sims(replace=True)
    wv_from_bin.save('data/kv.model')
model = KeyedVectors.load(wv_model_path, mmap='r')
