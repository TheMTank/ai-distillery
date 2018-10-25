""" Evaluate paper embeddings on downstream tasks """
import argparse
import glob
import numpy as np


from utils import load_word2vec_format

class Dataset(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SupervisedTask(object):
    def __init__(self, train, test):
        self.train = train
        self.test = test


class EmbeddingTransformer():
    def __init__(self, vocabulary, embedding, unk_vector=None):
        self.vocabulary = vocabulary
        self.embedding = np.asarray(embedding)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.asarray([self.embedding[self.vocabulary[x]] for x in X])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding-file", type=str,
                        help="Path to paper vectors in word2vec format")
    
    args = parser.parse_args()

    identifiers, embedding = load_word2vec_format(args.embedding_file)
    vocabulary = { identifier: idx for idx, identifier in enumerate(identifiers) }

    embed = EmbeddingTransformer(vocabulary, embedding)

    # Train a simple model on top of the embeddings







if __name__ == '__main__':
    import doctest
    doctest.testmod()
    main()
