import gensim
import argparse
from os import listdir
from os.path import isfile, join
from utils import clean_raw_data as cl
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_corpus(files, train_folder):
    for file in files:
        data = open(train_folder + file, "r").read().strip()
        # Normalize first, then remove stop words
        string = cl.normalize_text(data)
        string = cl.remove_stop_words(string)

        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(string), [file])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder',
                        help="Path to the folder that contains textual documents")
    parser.add_argument('-d', '--dimension', default="100",
                        help="Dimension of the desired embeddings", type=int)
    parser.add_argument('-mc', '--min_count', default="5",
                        help="Min number of occurrence of words", type=int)
    parser.add_argument('-o', '--output_file', default="output_embedding",
                        help="Output embedding file")

    args = parser.parse_args()
    train_folder = args.folder

    onlyfiles = [f for f in listdir(train_folder) if isfile(join(train_folder, f))]

    train_corpus = list(read_corpus(onlyfiles, train_folder))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=args.dimension, min_count=args.min_count)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(args.output_file)


if __name__ == '__main__':
    main()
