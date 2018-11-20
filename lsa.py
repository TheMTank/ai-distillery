""" Baselines for paper embeddings """
import glob
import sys
import pickle
import time
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

from utils.file_handling import identifier_from_path


"""
Use:
python lsa.py --lsa-bf-output data/paper_embeddings/lsa-600-components-250k-feats-for-IR --lsa-model-output data/models/lsa-tfidf-pipeline-600-dim.pkl --n-components 600
"""

start = time.time()

parser = argparse.ArgumentParser(description='Convert text files to BF format for visualisation and '
                                             'to a saved LSA model for vectorization of query text '
                                             'for IR')
parser.add_argument('-i', '--input-folder', type=str, help='Input folder path', required=True) # todo fix bug here
parser.add_argument('-o-bf', '--lsa-bf-output', type=str, help='Output LSA features in BF format', required=True)
parser.add_argument('-o-m', '--lsa-model-output', type=str, help='Output LSA model for IR', required=True)
parser.add_argument('-nc', '--n-components', default=100, type=int, help='Number of components for LSA')
parser.add_argument('-max-feats', '--max-features', default=50000, type=int, help='Number of max features for TFIDF')

args = parser.parse_args()

print(args)
# args.input_folder = '/home/beduffy/all_projects/arxiv-sanity-preserver/data/final_version_paper_txt_folder_v0_54797_papers_Nov4th/final_version_paper_txt/*.txt'
print('Input folder path: {}'.format(args.input_folder))
ALL_PAPER_PATHS = glob.glob(args.input_folder)
print(len(ALL_PAPER_PATHS))

with open('data/full_paper_id_to_title_dict.pkl', 'rb') as fhandle:
    ID2TITLE = pickle.load(fhandle)

LSA = Pipeline(
    [
        ("tfidf_vectorizer", TfidfVectorizer(input='filename', encoding='utf-8',
                                             decode_error='replace', strip_accents='unicode',
                                             lowercase=True, analyzer='word', stop_words='english',
                                             token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
                                             ngram_range=(1, 1), max_features=args.max_features,
                                             norm='l2', use_idf=True, smooth_idf=True,
                                             sublinear_tf=True, max_df=1.0, min_df=1)),
        ("svd", TruncatedSVD(n_components=args.n_components))
    ]
)

if __name__ == '__main__':
    print("Run {}-dim LSA on {} papers.".format(args.n_components, len(ALL_PAPER_PATHS)))
    LSA_EMBEDDING = LSA.fit_transform(ALL_PAPER_PATHS)
    print("Explained variance ratio sum:", LSA.named_steps.svd.explained_variance_ratio_.sum())
    # save_word2vec_format(OUTFILE, [identifier_from_path(p) for p in ALL_PAPERS], LSA_EMBEDDING)
    ALL_IDS = [identifier_from_path(p) for p in ALL_PAPER_PATHS]
    ALL_TITLES = [ID2TITLE.get(x, x) for x in ALL_IDS]
    EMBEDDING = {
        # DONE get titles to make this true ben format (ID2TITLE might be incomplete)
        'labels': ALL_TITLES,
        'embeddings': LSA_EMBEDDING
    }
    with open(args.lsa_bf_output, 'wb') as outfile:
        print('Saving LSA BF embedding to path: {}'.format(
            args.lsa_bf_output))
        pickle.dump(EMBEDDING, outfile)

    print('Shape of LSA_EMBEDDING: {}'.format(LSA_EMBEDDING.shape))

    with open(args.lsa_model_output, 'wb') as f:
        print('Saving LSA model pipeline object and ids + titles to path: {}'.format(
            args.lsa_model_output))
        LSA_IR_dict = {
            'model': LSA,
            'feats': LSA_EMBEDDING,
            'ids': ALL_IDS,
            'titles': ALL_TITLES
        }
        pickle.dump(LSA_IR_dict, f)

    print('Total time taken: {}'.format(time.time() - start))
