#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Baselines for paper embeddings """
import argparse
import glob
import os
import sys
import pickle

import gensim

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

from .file_handling import identifier_from_path
from .data_cleaning import normalize_text, remove_stop_words

##############################################################################
# LSA embedding methods
#######################

def lsa_main(args):
    """Runs lsa on a data directory

    :args: command line argument namespace
    """
    if args.outfile is None:
        os.makedirs(os.path.join("data", "tmp"), exist_ok=True)
        args.outfile = os.path.join("data", "tmp", f"lsa-{args.n_components}.pkl")
    print("LSA Embedding will be stored at:", args.outfile)
    lsa = Pipeline(
        [
            ("tfidf", TfidfVectorizer(input='filename', stop_words='english', max_features=50000)),
            ("svd", TruncatedSVD(n_components=args.n_components))
        ]
    )
    all_papers = glob.glob(os.path.join(args.data, "*"))
    print("Run {}-dim LSA on {} papers.".format(args.n_components, len(all_papers)))
    lsa_embedding = lsa.fit_transform(all_papers)
    print("Explained variance ratio sum:", lsa.named_steps.svd.explained_variance_ratio_.sum())
    # save_word2vec_format(OUTFILE, [identifier_from_path(p) for p in all_papers], LSA_EMBEDDING)
    labels = [identifier_from_path(p) for p in all_papers]

    if args.annotate is not None:
        with open(args.annotate, 'rb') as fhandle:
            id2title = pickle.load(fhandle)
        # Replace identifier labels with title labels (if possible)
        labels = [id2title.get(x, x) for x in labels]

    embedding_bf = {
        'labels': labels,
        'embeddings': lsa_embedding
    }

    with open(args.outfile, 'wb') as outfile:
        pickle.dump(embedding_bf, outfile)


def lsa_add_args(parser):
    parser.add_argument('data',
                        help="Path to dir containing full-texts")
    parser.add_argument('--annotate',
                        help="Path to pickled dict containing id to title mapping",
                        default=None)
    parser.add_argument('-n', '--n-components',
                        help="Number of dimensions", type=int,
                        default=300)
    parser.add_argument('-o', '--outfile', default=None,
                        help=("Destination to store lsa embeddings in ben format. "
                              "Default is 'data/tmp/lsa-{n_components}.pkl'"))


##############################################################################
# Doc2vec embedding methods
###########################

def read_corpus(files, train_folder):
    for file in files:
        data = open(train_folder + file, "r").read().strip()
        # Normalize first, then remove stop words
        string = normalize_text(data)
        string = remove_stop_words(string)

        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(string), [file])

def doc2vec_main(args):
    """Run doc2vec on a bunch of documents

    :args: argument namespace

    """
    train_folder = args.folder

    onlyfiles = [f for f in os.listdir(train_folder) \
                 if os.path.isfile(os.path.join(train_folder, f))]

    train_corpus = list(read_corpus(onlyfiles, train_folder))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=args.dimension, min_count=args.min_count)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(args.output_file)

def doc2vec_add_args(parser):
    parser.add_argument('-f', '--folder',
                        help="Path to the folder that contains textual documents")
    parser.add_argument('-d', '--dimension', default="100",
                        help="Dimension of the desired embeddings", type=int)
    parser.add_argument('-mc', '--min_count', default="5",
                        help="Min number of occurrence of words", type=int)
    parser.add_argument('-o', '--output_file', default="output_embedding",
                        help="Output embedding file")

