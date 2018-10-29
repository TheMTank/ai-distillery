""" Baselines for paper embeddings """
import glob
import sys
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

from utils import save_word2vec_format, identifier_from_path

ALL_PAPERS = glob.glob("data/txt/*")

N_COMPONENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUTFILE = "embeddings/lsa-{}.w2v.txt".format(N_COMPONENTS)

LSA = Pipeline(
    [
        ("tfidf", TfidfVectorizer(input='filename', stop_words='english', max_features=50000)),
        ("svd", TruncatedSVD(n_components=N_COMPONENTS))
    ]
)


if __name__ == '__main__':
    print("Run {}-dim LSA on {} papers.".format(N_COMPONENTS, len(ALL_PAPERS)))
    LSA_EMBEDDING = LSA.fit_transform(ALL_PAPERS)
    print("Explained variance ratio sum:", LSA.named_steps.svd.explained_variance_ratio_.sum())
    # save_word2vec_format(OUTFILE, [identifier_from_path(p) for p in ALL_PAPERS], LSA_EMBEDDING)
    embedding = {
        'labels': [identifier_from_path(p) for p in ALL_PAPERS],
        'vectors': LSA_EMBEDDING
    }
    with open(OUTFILE, 'w') as outfile:
        pickle.dump(embedding, outfile)
