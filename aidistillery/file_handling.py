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


class BF:
    def __init__(self, labels=list(), embeddings=list()):
        self.labels = labels
        self.embeddings = embeddings

    def load_from_gensim(self, model):

        self.labels = list(model.wv.vocab.keys())
        self.embeddings = np.concatenate([model[word].reshape(1, -1) for word in self.labels], axis=0)

    def save(self, output_location):
        """
        Save bf model using pickle
        :param output_location:
        :return:
        """
        bf_model = {'labels': self.labels, 'embeddings': self.embeddings}

        with open(output_location, 'wb') as handle:
            pickle.dump(bf_model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def gensim_to_bf(model):
    """
    Convers a gensim model to bf format and save pickled file
    :param model:
    :return:
    """
    vocab = list(model.wv.vocab.keys())

    embeddings_array = np.concatenate([model[word].reshape(1, -1) for word in vocab], axis=0)

    bf = BF(vocab, embeddings_array)
    return bf
