from pprint import pprint

import numpy as np
import pandas as pd


def check_labels():
    df = pd.read_csv('train.csv')
    print(np.unique(df['emotions'], return_counts=True))


def check_results():
    filename = 'lr_plot_data.pkl'
    with open(f'models/{filename}', 'rb') as f:
        data = pd.read_pickle(f)
        pprint(data)


def main():
    # check_labels()
    check_results()


if __name__ == '__main__':
    main()
