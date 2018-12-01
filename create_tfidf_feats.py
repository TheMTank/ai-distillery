""" Baselines for paper embeddings """
import glob
import sys
import pickle
import time
import argparse

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.file_handling import identifier_from_path


"""
python create_tfidf_feats.py --tfidf-model-output data/models/tfidf-200k-feats-IR-object.pkl --max-features 200000
python create_tfidf_feats.py --tfidf-model-output data/models/tfidf-50k-feats-IR-object.pkl
python create_tfidf_feats.py --tfidf-model-output data/models/tfidf-25k-feats-IR-object.pkl --max-features 25000
"""

start = time.time()

parser = argparse.ArgumentParser(description='Convert text files to BF format for visualisation and '
                                             'to a saved TFIDF model for vectorization of query text '
                                             'for IR')
parser.add_argument('-i', '--input-folder', type=str, help='Input folder path', required=False) # todo should be True
parser.add_argument('-o-m', '--tfidf-model-output', type=str, help='Output TFIDF model for IR', required=True)
parser.add_argument('-max-feats', '--max-features', default=50000, type=int, help='Number of max features for TFIDF')

args = parser.parse_args()

print(args)
args.input_folder = '/home/beduffy/all_projects/arxiv-sanity-preserver/data/final_version_paper_txt_folder_v0_54797_papers_Nov4th/final_version_paper_txt/*.txt'
print('Input folder path: {}'.format(args.input_folder))
ALL_PAPER_PATHS = glob.glob(args.input_folder)
print(len(ALL_PAPER_PATHS))

with open('data/full_paper_id_to_title_dict.pkl', 'rb') as fhandle:
    ID2TITLE = pickle.load(fhandle)
with open('data/full_paper_id_to_abstract_dict.pkl', 'rb') as fhandle:
    ID2ABSTRACT = pickle.load(fhandle)

vocab = list(set(pd.read_csv('data/latest_tfidf_top_features.csv')['feature_name']))

tfidf_vectorizer = TfidfVectorizer(input='filename', encoding='utf-8',
                                   decode_error='replace', strip_accents='unicode',
                                   lowercase=True, analyzer='word', stop_words='english',
                                   token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
                                   ngram_range=(1, 1),
                                   #max_features=args.max_features,
                                   vocabulary=vocab,
                                   norm='l2', use_idf=True, smooth_idf=True,
                                   sublinear_tf=True, max_df=1.0, min_df=40)

if __name__ == '__main__':
    # ALL_PAPER_PATHS = ALL_PAPER_PATHS[:1000]
    print("Run {}-feat TFIDF on {} papers.".format(args.max_features, len(ALL_PAPER_PATHS)))

    tfidf_feats = tfidf_vectorizer.fit_transform(ALL_PAPER_PATHS)
    ALL_IDS = [identifier_from_path(p) for p in ALL_PAPER_PATHS]
    ALL_TITLES = [ID2TITLE.get(x, x) for x in ALL_IDS]
    ALL_ABSTRACTS = [ID2ABSTRACT.get(x, x) for x in ALL_IDS]

    print('Shape of tfidf_feats: {}'.format(tfidf_feats.shape))

    with open(args.tfidf_model_output, 'wb') as f:
        print('Saving TFIDF model pipeline object and ids + titles to path: {}'.format(
            args.tfidf_model_output))
        TFIDF_IR_dict = {
            'model': tfidf_vectorizer,
            'feats': tfidf_feats,
            'ids': ALL_IDS,
            'titles': ALL_TITLES,
            'abstracts': ALL_ABSTRACTS
        }
        pickle.dump(TFIDF_IR_dict, f)

    indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
    feature_names = tfidf_vectorizer.get_feature_names()
    top_n = 50
    top_features = [feature_names[i] for i in indices[:top_n]]
    print('top_features top 50:')
    print(top_features)

    all_features_sorted = [feature_names[i] for i in indices]
    idfs = tfidf_vectorizer.idf_[indices]

    df_features_sorted = pd.DataFrame({'feature_name': all_features_sorted, 'idf': idfs})
    df_features_sorted.to_csv('data/latest_tfidf_top_features.csv')

    print('Total time taken: {}'.format(time.time() - start))
