import os
import datetime
import pickle
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

def generate_file_name(type, dimension, dataset):
    """
    :param type:
    :param dimension:
    :param dataset:
    :return:

    >>> generate_file_name("doc2vec", 100, "arxivNov")
    'type_doc2vec#dim_100#dataset_arxivNov#time_2018-11-09T18:30:05.173235'
    """

    return "type_" + type + "#dim_" + str(dimension) + "#dataset_" + dataset + "#time_" \
           + datetime.datetime.now().isoformat()


def gensim_to_bf(model):
    """
    Convers a gensim model to bf format and save pickled file
    :param model:
    :return:
    """
    vocab = list(model.wv.vocab.keys())

    embeddings_array = np.concatenate([model[word].reshape(1, -1) for word in vocab], axis=0)

    embeddings_bf = {'labels': vocab, 'embeddings': embeddings_array}
    return embeddings_bf

def save_bf_to_pickle(bf_model, output_location):
    """
    Save bf model using pickle
    :param bf_model:
    :param folder_location:
    :return:
    """
    with open(output_location, 'wb') as handle:
        pickle.dump(bf_model, handle, protocol=pickle.HIGHEST_PROTOCOL)



###########################################################
# Deprecated. Old methods to save and load word2vec-format.
# Please use Ben-format instead.

def save_word2vec_format(path, words, vectors):
    """ Saves an embedding, words must have corresponding indices """
    vectors = np.asarray(vectors)
    num_embeddings, embedding_dim = len(words), vectors.shape[1]
    with open(path, 'w') as outfile:
        print("{} {}".format(num_embeddings, embedding_dim), file=outfile)
        for idx, word in enumerate(words):
            line = "{} {}".format(word, ' '.join(map(str, vectors[idx])))
            print(line, file=outfile)


def load_word2vec_format(path):
    """ Loads an embedding in word2vec format """
    words = []
    with open(path, 'r') as fhandle:
        num_embeddings, embedding_dim = tuple(map(int, next(fhandle).strip().split(' ')))
        embedding = np.empty((num_embeddings, embedding_dim))
        for idx, line in enumerate(fhandle):
            word, *numbers = line.strip().split(' ')
            words.append(word)
            embedding[idx, :] = np.array(numbers)

    return words, embedding

###########################################################
