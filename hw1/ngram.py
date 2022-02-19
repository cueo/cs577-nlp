import numpy as np
import pandas as pd


class NGram:
    def __init__(self, n):
        self.n = n
        self.grams_dict = {}
        self.grams = []

    def fit_transform(self, series, k_best=None):
        for _, row in series.items():
            grams = self.text_to_grams(row)
            for gram in grams:
                self.grams_dict[gram] = self.grams_dict.get(gram, 0) + 1
        self.select_k_best(k_best)

        vectors = self.transform(series)
        return vectors

    def transform(self, series):
        # print('Transforming text to n-gram vectors.')
        vectors = np.array([np.zeros(len(self.grams))])
        total = series.shape[0]
        for i, row in series.items():
            # if i % 100 == 0:
            #     print(f'Processed {i}/{total} rows.')
            row_grams = self.text_to_grams(row)
            vector = np.array([1 if gram in row_grams else 0 for gram in self.grams])
            vectors = np.append(vectors, [vector], axis=0)
        vectors = np.delete(vectors, 0, axis=0)
        # vectors = pd.Series(vectors.tolist(), index=series.index)
        print('Transformation completed.')
        return vectors

    def select_k_best(self, k_best):
        total = len(self.grams_dict)
        # print(f'Selecting top {k_best} out of {total} features.')
        if k_best is None or k_best >= total:
            self.grams = list(self.grams_dict.keys())
        else:
            self.grams = sorted(self.grams_dict, key=self.grams_dict.get, reverse=True)[:k_best]

    def text_to_grams(self, row):
        words = row.split()
        length = len(words)
        if self.n == 1:
            row_grams = [tuple([words[i]]) for i in range(length)]
        else:
            row_grams = [tuple(words[i:i + self.n]) for i in range(length - self.n + 1)]
        return row_grams
