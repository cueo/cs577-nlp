import torch
from torch import nn

DIMENSION = 128

torch.manual_seed(42)


def get_word_to_idx(sentences):
    # create dictionary of unique words in train_sents
    word_to_idx = {
        'unk': 0
    }
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    return word_to_idx


def get_word_to_idx_from_model(model):
    return model.key_to_index


class RandomWordEmbeddings:
    def __init__(self, word_to_idx, embedding_dim):
        self.word_to_idx = word_to_idx
        self.embedding_dim = embedding_dim

    def embeddings(self):
        return nn.Embedding(len(self.word_to_idx), embedding_dim=self.embedding_dim)


class Word2VecEmbeddings:
    def __init__(self, model):
        self.model = model

    def embeddings(self):
        weights = torch.tensor(self.model.vectors, dtype=torch.float)
        return nn.Embedding.from_pretrained(weights)


def word_embeddings(option, train_sents, model):
    sentences = train_sents
    word_to_idx = get_word_to_idx(sentences)
    if option == 1:
        embeds = RandomWordEmbeddings(word_to_idx, DIMENSION).embeddings()
    else:
        # model = wv_from_bin
        # model.build_vocab(sentences, update=True)
        word_to_idx = get_word_to_idx_from_model(model)
        embeds = Word2VecEmbeddings(model).embeddings()
    words = {word for sentence in sentences for word in sentence if word in word_to_idx}
    word_to_embedding = {
        word:
            embeds(torch.tensor(word_to_idx[word], dtype=torch.long)) for word in words
    }
    word_to_embedding['unk'] = embeds(torch.tensor(word_to_idx['unk'], dtype=torch.long))
    return embeds, word_to_idx, word_to_embedding


def tag_embeddings():
    tag_to_idx = {'<PAD>': 0, 'O': 1, 'T-POS': 2, 'T-NEG': 3, 'T-NEU': 4}
    tag_to_embedding = {
        tag:
            torch.tensor([tag_to_idx[tag]], dtype=torch.long) for tag in tag_to_idx
    }
    return tag_to_idx, tag_to_embedding


def sentence_embedding(embeds, sentence, word_to_idx, max_len):
    idxs = []
    for word in sentence:
        if word not in word_to_idx:
            word = '<PAD>'
        idxs.append(word_to_idx[word])
    idxs = idxs + [word_to_idx['<PAD>']] * (max_len - len(idxs))
    embedding = embeds(torch.tensor(idxs, dtype=torch.long))
    return embedding


def build_sentence_embeddings(sentences, embeds, word_to_idx):
    max_len = max([len(sentence) for sentence in sentences])
    embeddings = []
    for sentence in sentences:
        embedding = sentence_embedding(embeds, sentence, word_to_idx, max_len)
        embeddings.append(embedding)
    return embeddings
