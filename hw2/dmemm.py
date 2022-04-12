import time

import numpy as np

import torch
import torch.nn.functional as F

from torch import nn, optim, Tensor


class DeepMarkovModel(nn.Module):
    def __init__(self, embeddings, hidden_dim, tag_size):
        super(DeepMarkovModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.embeddings = embeddings

        self.input_dim = embeddings.embedding_dim + 1  # +1 for the prev_tag
        self.layer1 = nn.Linear(self.input_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)

    def forward(self, context: Tensor):
        # embeds = self.embeddings(sentence)
        # hiddens, _ = self.gru(embeds.view(1, len(context)))
        hiddens = self.layer1(context)
        activations = F.relu(hiddens)
        tag_space = self.hidden2tag(activations)
        tag_scores = F.log_softmax(tag_space, dim=0)
        return tag_scores


class GRUMarkovModel(nn.Module):
    def __init__(self, hidden_dim, embeddings, tag_size):
        super(GRUMarkovModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings

        self.input_size = embeddings.embedding_dim + 1  # +1 for the prev_tag
        self.gru = nn.GRU(self.input_size, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(2 * hidden_dim, tag_size)

    def forward(self, context: Tensor):
        # hiddens, _ = self.gru(context)
        hiddens, _ = self.gru(context.view(1, len(context), -1))
        tag_space = self.hidden2tag(hiddens.view(len(context), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def prepare_context(sentences, tags, word_to_embedding, tag_to_embedding):
    sequences = []
    tag_sequences = []
    prev_tag = '<PAD>'
    for sentence, tag in zip(sentences, tags):
        for word, _tag in zip(sentence, tag):
            word = word if word in word_to_embedding else 'unk'
            sequences.append(
                torch.cat([word_to_embedding[word], tag_to_embedding[prev_tag]], dim=0))
            tag_sequences.append(tag_to_embedding[_tag])
            prev_tag = _tag
    return torch.stack(sequences), torch.stack(tag_sequences)


def _train(model, sentences, tags, word_to_embeddings, tag_to_embeddings, epochs, lr):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    sequences, tag_sequences = prepare_context(sentences, tags, word_to_embeddings, tag_to_embeddings)
    tags_actual = tag_sequences.reshape(len(tag_sequences))
    for epoch in range(epochs):
        time_start = time.time()
        i = 0
        batch_loss = []
        # while i <= len(sequences) // batch_size:
        #     start = i * batch_size
        #     end = (i + 1) * batch_size
        #     # print(f'Batch: Processing from {start} to {end}')
        #     contexts = sequences[start:end]
        #     tags_actual = tag_sequences[start:end]
        #     i += 1
        #     optimizer.zero_grad()
        #     tag_scores = model(contexts)
        #     loss = loss_fn(tag_scores, tags_actual.reshape(len(tags_actual)))
        #     loss.backward(retain_graph=True)
        #     batch_loss.append(loss.item())
        #     optimizer.step()
        # print(f'Epoch: {epoch}, Loss: {np.mean(batch_loss)}')
        optimizer.zero_grad()
        tag_scores = model(sequences)
        loss = loss_fn(tag_scores, tags_actual)
        loss.backward(retain_graph=True)
        optimizer.step()
        time_del = time.time() - time_start
        print(f'Epoch: {epoch}, Loss: {np.mean(loss.item())} in {time_del:.2f} seconds')
    return model


def train(sentences, tags, embeddings, word_to_embeddings, tag_to_embeddings,
          hidden_dim, lr, epochs):
    model = DeepMarkovModel(embeddings, hidden_dim, len(tag_to_embeddings))
    return _train(model, sentences, tags, word_to_embeddings, tag_to_embeddings, epochs, lr)


def prepare_sequence(sentence, embeddings, word_to_idx, tags, tag_to_embeddings):
    sequences = []
    for i in range(len(sentence)):
        word = sentence[i] if sentence[i] in word_to_idx else 'unk'
        word_embed = embeddings(torch.tensor(word_to_idx[word], dtype=torch.long))
        tag = tag_to_embeddings[tags[i]]
        sequence = torch.cat([word_embed, tag], dim=0)
        sequences.append(sequence)
    return torch.stack(sequences)


def train_gru(sentences, tags, embeddings, word_to_embeddings, tag_to_embeddings,
              hidden_dim, lr, epochs):
    model = GRUMarkovModel(hidden_dim, embeddings, len(tag_to_embeddings))
    return _train(model, sentences, tags, word_to_embeddings, tag_to_embeddings, epochs, lr)
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # for epoch in range(epochs):
    #     time_start = time.time()
    #     batch_loss = []
    #
    #     for sentence, tag in zip(sentences, tags):
    #         optimizer.zero_grad()
    #         context = prepare_sequence(sentence, embeddings, word_to_idx, tag, tag_to_embeddings)
    #         tag_scores = model(context)
    #         tag_actual = torch.tensor([tag_to_embeddings[i] for i in tag], dtype=torch.long)
    #         loss = loss_fn(tag_scores, tag_actual)
    #         loss.backward()
    #         optimizer.step()
    #         batch_loss.append(loss.item())
    #     time_del = time.time() - time_start
    #     print(f'Epoch: {epoch}, Loss: {np.mean(batch_loss)} in {time_del:.2f} s')
    # return model


def get_embedding(word, word_embeddings, word_to_idx):
    if word not in word_to_idx:
        word = 'unk'
    return word_embeddings(torch.tensor(word_to_idx[word], dtype=torch.long))


def forward(context, model):
    if isinstance(model, GRUMarkovModel):
        context = context.reshape(1, context.shape[0])
        tag_scores = model(context)[0]
    else:
        tag_scores = model(context)
    return tag_scores[1:]


def viterbi(sentences, model, word_embeddings, word_to_idx, tag_to_embeddings):
    tags = ['O', 'T-POS', 'T-NEG', 'T-NEU']
    predictions = []
    for idx, sentence in enumerate(sentences):
        viterbi_matrix = np.ones((len(tags), len(sentence))) * -np.inf
        backpointer_matrix = np.empty((len(tags), len(sentence)), dtype=int)
        prev_tag = '<PAD>'
        word = get_embedding(sentence[0], word_embeddings, word_to_idx)
        context = torch.cat([word, tag_to_embeddings[prev_tag]], dim=0)
        tag_scores = forward(context, model)
        viterbi_matrix[:, 0] = tag_scores.detach().numpy()
        backpointer_matrix[:, 0] = 0
        for w in range(1, len(sentence)):
            for pt, prev_tag in enumerate(tags):
                word = get_embedding(sentence[w], word_embeddings, word_to_idx)
                context = torch.cat([word, tag_to_embeddings[prev_tag]], dim=0)
                tag_scores = forward(context, model)
                for t, tag in enumerate(tags):
                    new_score = tag_scores[t] + viterbi_matrix[pt, w - 1]
                    if new_score > viterbi_matrix[t, w]:
                        viterbi_matrix[t, w] = new_score
                        backpointer_matrix[t, w] = pt
        '''
        for i, word in enumerate(sentence):
            if i == 0:
                continue
            word_embedding = word_to_embeddings[word]
            for j, tag in enumerate(tags):
                if j == 0:
                    continue
                tag_embedding = tag_to_embeddings[prev_tag]
                context = torch.cat([word_embedding, tag_embedding], dim=0)
                tag_scores = model(context)
                tag_scores = tag_scores.detach().numpy()
                for k, tag_score in enumerate(tag_scores):
                    if viterbi_matrix[k, i] < viterbi_matrix[j, i - 1] + tag_score:
                        viterbi_matrix[k, i] = viterbi_matrix[j, i - 1] + tag_score
                        backpointer_matrix[k, i] = j
                prev_tag = tags[np.argmax(tag_scores)]
        '''

        prediction = np.empty(len(sentence), dtype='<U8')
        last_tag_idx = np.argmax(viterbi_matrix[:, -1])
        prediction[-1] = tags[last_tag_idx]
        for i in reversed(range(1, len(sentence))):
            prediction[i - 1] = tags[backpointer_matrix[last_tag_idx, i]]
            last_tag_idx = backpointer_matrix[last_tag_idx, i]
        predictions.append(prediction)

    return predictions


def evaluate(predictions, tags):
    tp, fp, fn = 0, 0, 0
    for tag, prediction in zip(tags, predictions):
        if prediction == tag:
            if prediction != 'O':  # T-XX T-XX
                tp += 1
            else:  # O O
                pass
        else:
            if prediction != 'O':
                if tag == 'O':  # O T-XX
                    fp += 1
                else:  # T-XX T-YY
                    fp += 1
                    fn += 1
            else:  # T-XX O
                fn += 1
    precision = 0
    if tp + fp != 0:
        precision = tp * 100 / (tp + fp)
    recall = 0
    if tp + fn != 0:
        recall = tp * 100 / (tp + fn)
    f1 = 0
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def write_predictions(option, predictions, test_path):
    if test_path is None:
        return
    with open(test_path, 'r') as f:
        sentences = f.read().splitlines()
    filepath = f'predictions_{option}.txt'
    tagged_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i].split('####')[0]
        prediction = predictions[i]
        tagged_sentence = sentence + '####'
        words = sentence.split(' ')
        for j in range(len(words)):
            word = words[j]
            tag = prediction[j]
            tagged_sentence += f'{word}={tag} '
        tagged_sentence = tagged_sentence[:-1] + '\n'
        tagged_sentences.append(tagged_sentence)
    with open(filepath, 'w') as f:
        f.writelines(tagged_sentences)


def predict(option, sentences, tags, model, word_embeddings, word_to_idx, tag_to_embeddings, test_path=None):
    predictions = viterbi(sentences, model, word_embeddings, word_to_idx, tag_to_embeddings)
    write_predictions(option, predictions, test_path)
    predictions = np.array([p for preds in predictions for p in preds])
    tags = np.array([t for tag in tags for t in tag])
    precision, recall, f1 = evaluate(predictions, tags)
    print(f'\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}')
    return precision, recall, f1
