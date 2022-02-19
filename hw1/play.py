import numpy as np
import pandas as pd


def main():
    df = pd.read_csv('train.csv')
    print(np.unique(df['emotions'], return_counts=True))


if __name__ == '__main__':
    main()
