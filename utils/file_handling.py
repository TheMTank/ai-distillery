import os
import datetime

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