import re
import pickle

path_to_gensim_vectors_pkl = 'data/word_embeddings/gensim_vectors.pkl'
with open(path_to_gensim_vectors_pkl, 'rb') as handle:
    print('Attempting to open file at: ', path_to_gensim_vectors_pkl)
    embeddings_object = pickle.load(handle, encoding='latin1')
    vocab = embeddings_object['labels']
    embeddings_array = embeddings_object['embeddings']
    embeddings_dict = {vocab[i]: embeddings_array[i] for i in range(len(vocab))}
    print('Finished reading file and creating embeddings dictionary')
    print('Len of vocab: {}. Shape of embeddings: {}'.format(len(vocab), embeddings_array.shape))

    new_vocab = []
    for word in vocab:
        word = re.sub("\S*\d\S*", "", word).strip()
        word = re.sub('[^A-Za-z]', '', word).strip()
        if len(word) != 0:
            new_vocab.append(word)

    print('vocab size after cleaning with regex: {}'.format(len(new_vocab)))

    print('lowercase set unique size: {}'.format(len(set([x.lower() for x in new_vocab]))))
