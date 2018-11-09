""" Baselines for paper embeddings """
import glob
import sys
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

from utils.file_handling import identifier_from_path

ALL_PAPERS = glob.glob("data/txt/*")

N_COMPONENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUTFILE = "embeddings/lsa-{}.pkl".format(N_COMPONENTS)

with open('data/full_paper_id_to_title_dict.pkl', 'rb') as fhandle:
    ID2TITLE = pickle.load(fhandle)

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
    ALL_IDS = [identifier_from_path(p) for p in ALL_PAPERS]
    EMBEDDING = {
        # DONE get titles to make this true ben format (ID2TITLE might be incomplete)
        'labels': [ID2TITLE.get(x, x) for x in ALL_IDS],
        'embeddings': LSA_EMBEDDING
    }
    with open(OUTFILE, 'wb') as outfile:
        pickle.dump(EMBEDDING, outfile)
