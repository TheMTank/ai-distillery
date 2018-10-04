""" Utilities for the AI distillery """
import os
import numpy as np

def identifier_from_path(path):
    """
    Gets the arxiv identifier from a path

    >>> identifier_from_path("0507037v3.pdf.txt")
    '0507037v3'
    >>> identifier_from_path("0705.4676v8.pdf.txt")
    '0705.4676v8'
    >>> identifier_from_path("data/txt/0705.4676v8.pdf.txt")
    '0705.4676v8'
    """
    basename = os.path.basename(path)
    return os.path.splitext(os.path.splitext(basename)[0])[0]

def save_word2vec_format(path, words, vectors):
    """ Saves an embedding, words must have corresponding indices """
    vectors = np.asarray(vectors)
    num_embeddings, embedding_dim = len(words), vectors.shape[1]
    with open(path, 'w') as outfile:
        print("{} {}".format(num_embeddings, embedding_dim), file=outfile)
        for idx, word in enumerate(words):
            line = "{} {}".format(word, ' '.join(map(str, vectors[idx])))
            print(line, file=outfile)


def load_word2vec_format(path, dtype=np.float64):
    """ Loads an embedding in word2vec format """
    words = []
    with open(path, 'r') as fhandle:
        num_embeddings, embedding_dim = tuple(map(int, next(fhandle).strip().split(' ')))
        embedding = np.empty((num_embeddings, embedding_dim), dtype)
        for idx, line in enumerate(fhandle):
            word, *numbers = line.strip().split(' ')
            words.append(word)
            embedding[idx, :] = np.array(numbers)

    return words, embedding


if __name__ == '__main__':
    import doctest
    doctest.testmod()
