from time import time

from data import load_seq_data


if __name__ == '__main__':
    start = time()
    load_seq_data("../data/small_data_samples.txt", 3, 7)
    end = time()
    print(end - start)