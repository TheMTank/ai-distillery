"""
embed.py (Executable)

Generate embeddings using word2vec from a text collection

Example: `python3 embed.py -f data/text -o data/saved_embedding -d 100`
"""

import argparse
from gensim.models import word2vec
from utils import clean_raw_data as cl

def load_and_process(file_name, min_length):
    """
    Arguments
        ---------
        file_name: file that contains text from which we want to generate embeddings
        min_length: required min length of sentences to be considered

        Returns
        -------
        List of list of tokens for gensim.word2vec input
    """

    return cl.clean_raw_text_from_file(file_name, min_length=min_length)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', default="text_to_embed.txt",
                        help="Text file") 
    parser.add_argument('-d', '--dimension', default="100",
                        help="Dimension of the desired embeddings", type=int)
    parser.add_argument('-w', '--window', default="5",
                        help="Size of the window", type=int)
    parser.add_argument('-mc', '--min_count', default="5",
                        help="Min number of occurrence of words", type=int)
    parser.add_argument('-ml', '--min_length', default="200",
                        help="Min number of chars for a sentence", type=int)
    parser.add_argument('-it', '--iterations', default="5",
                    help="Number of iteration for learning the embeddings", type=int)
    parser.add_argument('-ws', '--workers', default="2",
                        help="Number of workers for this task",     type=int)
    parser.add_argument('-o', '--output_file', default="output_embedding",
                        help="Output embedding file")

    args = parser.parse_args()

    print("Creating Embeddings with:", args.dimension, "dimensions", args.window, "window", args.min_count, "min_count")
    print("Sentences with less than", args.min_length, "chars will be removed")

    sentences = load_and_process(args.file_name, args.min_length)

    model = word2vec.Word2Vec(sentences, 
        size=args.dimension,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers, 
        sg=0, 
        iter=args.iterations)

    model.save(args.output_file)


if __name__ == '__main__':
    main()
