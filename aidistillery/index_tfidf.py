#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer

from aidistillery.file_handling import identifier_from_path


def tfidf_main(args):
    print('Beginning tfidf_main(). With args: {}'.format(args))
    raw_documents = glob.glob(os.path.join(args.datadir, '*'))

    vect_args = {
        'input': 'filename',
        'stop_words': 'english',
    }

    if args.hash:
        if args.n_features is None:
            args.n_features = 2 ** 20  # sklearn default for hashingvectorizer
        else:
            print("Limiting vocab with hashing vectorizer cause hashes to collide")

        vect = HashingVectorizer(n_features=args.n_features, norm=None, **vect_args)
    else:
        vect = CountVectorizer(max_features=args.n_features, **vect_args)                               

    tfidf = TfidfTransformer(norm='l2')

    X = vect.fit_transform(raw_documents)

    X = tfidf.fit_transform(X)
    ids = [identifier_from_path(d) for d in raw_documents]

    print('Creating vectorizer, tfidf vector and index2identifier pickle objects')
    os.makedirs(args.outpath, exist_ok=True)
    with open(os.path.join(args.outpath, "vectorizer.pkl"), 'wb') as fhandle:
        pickle.dump(vect, fhandle)

    with open(os.path.join(args.outpath, "tfidf_sparse.pkl"), 'wb') as fhandle:
        pickle.dump(X, fhandle)

    with open(os.path.join(args.outpath, "index2identifier.pkl"), 'wb') as fhandle:
        pickle.dump(ids, fhandle)

def tfidf_add_args(parser):
    parser.add_argument("-d", "--datadir", required=True,
                        help="Path to txt file root")
    parser.add_argument("-o", "--outpath", required=True,
                        help="Path to output directory")
    parser.add_argument("-n", "--n-features", default=None, type=int,
                        help="Use that many features (default unlimited)")
    parser.add_argument("--hash", action='store_true', default=False,
                        help="Use (stateless) hashing vectorizer")


if __name__ == "__main__":
    import argparse
    PARSER = argparse.ArgumentParser()
    tfidf_add_args(PARSER)
    ARGS = PARSER.parse_args()
    tfidf_main(ARGS)
