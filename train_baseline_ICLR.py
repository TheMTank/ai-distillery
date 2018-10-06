import sys
import random
import json
import re
import time
import pickle

from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import pandas as pd

from stopwords import stopwords

# np.set_printoptions(threshold=np.nan)

def get_data(save_data=False, fp_output='data/all_ICLR_submissions/preprocessed_data/bag_of_words.pkl'):
    df = pd.read_csv('data/all_ICLR_submissions/ICLR_binary_classification.csv')
    print(df.head())
    print(df['title'])

    # print(np.sum(df['accepted']))
    # print(np.sum(df['accepted'] == np.array([random.choice([0, 1]) for x in range(1000)])))
    # print(np.sum(df['accepted'] == np.array([random.choice([0, 1]) for x in range(1000)])))
    # print(df.shape)

    # titles_word_split = [x.lower().split(' ') for x in df['title']]
    # print(titles_word_split)
    # all_words = [word for arr in titles_word_split for word in arr]
    # print(all_words)
    # unique_words = list(set(all_words))
    # num_unique_words = len(unique_words)
    # print(num_unique_words)
    # word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
    # idx_to_word = {idx: word for idx, word in enumerate(unique_words)}

    # with open('data/all_ICLR_submissions/all_ICLR_submission_id_to_title.json') as f:
    #     all_ICLR_accepted_id_to_title = json.load(f)
    #
    # all_ICLR_accepted_id_to_title['B1QRgziT-']

    start = time.time()
    all_doc_words = []
    for paper_id in df['paper_id']:
        fp = 'data/all_ICLR_submissions/txt/pdf?id={}.pdf.txt'.format(paper_id)
        with open(fp) as f:
            lines = [line.lower().strip() for line in f.readlines()]
            words_in_doc = [word for line in lines for word in line.split(' ') if word not in stopwords]

            cleaned_words_in_doc = []
            for word in words_in_doc:

                word = re.sub("\S*\d\S*", "", word).strip()
                word = re.sub('[^A-Za-z]', '', word).strip()
                if len(word) != 0:
                    cleaned_words_in_doc.append(word)

            # all_doc_words.append(words_in_doc)
            all_doc_words.append(cleaned_words_in_doc)

    all_words = [word for doc in all_doc_words for word in doc]
    # print(all_words)
    unique_words = list(set(all_words))
    num_unique_words = len(unique_words)
    print('Num unique words: ', num_unique_words)
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
    idx_to_word = {idx: word for idx, word in enumerate(unique_words)}

    print('Time taken to load data, clean and create mappings: {}'.format(time.time() - start))

    # for titles
    # bag_of_word_vecs = []
    # for title_split in titles_word_split:
    #     bag_of_words = np.zeros((1, num_unique_words))
    #     for word in title_split:
    #         bag_of_words[0, word_to_idx[word]] += 1 # todo check if it works
    #
    #     bag_of_word_vecs.append(bag_of_words)

    # for entire doc text
    print('Creating bag of words features')
    bag_of_word_vecs = []
    for idx, words_in_doc in enumerate(all_doc_words):
        bag_of_words = np.zeros((1, num_unique_words))
        for word in words_in_doc:
            bag_of_words[0, word_to_idx[word]] += 1 # todo check if it works

        bag_of_word_vecs.append(bag_of_words)

        if (idx + 1) % 100 == 0:
            print('{}/{}'.format(idx + 1, len(all_doc_words)))

    print('Finished creating bag of words, turning into design matrix')
    bag_of_word_vecs = np.concatenate(bag_of_word_vecs)
    # print(bag_of_word_vecs[0:30])
    print(bag_of_word_vecs.shape)

    if save_data:
        with open(fp_output, 'wb') as f:
            save_object = {
                'X': bag_of_word_vecs,
                'y': df['accepted'],
                'unique_words': unique_words
            }
            pickle.dump(save_object, f)

    return bag_of_word_vecs, df['accepted']

def get_preprocessed_data(fp='data/all_ICLR_submissions/preprocessed_data/bag_of_words.pkl'):
    with open(fp, 'rb') as f:
        dataset_obj = pickle.load(f)
        X = dataset_obj['X']
        y = dataset_obj['y']
        unique_words = dataset_obj['unique_words']

    return X, y, unique_words

# X, y = get_data(save_data=True)
# sys.exit()
X, y, unique_words = get_preprocessed_data()

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=42)
print('X shape: {}. y shape: {}. Num accepted: {}'.format(X.shape, y.shape, y.sum()))
print('X_train shape: {}. y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_test shape: {}. y_test shape: {}'.format(X_test.shape, y_test.shape))

print('Beginning training')
all_clfs = {
    'RFC 100 tree': RandomForestClassifier(n_estimators=100,
                             #max_depth=2,
                              random_state=0), # 0.7
    # 'MultinomialNB': MultinomialNB(), # 0.61
    # 'KNeighborsClassifier': KNeighborsClassifier(3), # 0.535
    # 'SVC linear': SVC(kernel="linear", C=0.025), # 0.725
    # 'SVC rbf': SVC(gamma=2, C=1), # 0.46
    # 'GaussianProcessClassifier': GaussianProcessClassifier(1.0 * RBF(1.0)), # 0.825
    # 'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=5), # 0.83
    'LogisticRegression': LogisticRegression(), # 0.73
    # 'RandomForestClassifier': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), # 0.46
    # 'MLPClassifier': MLPClassifier(alpha=1),  #0.69
    # 'AdaBoostClassifier': AdaBoostClassifier(), # 0.825
    # 'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis() # 0.525
}

# Fixed train and test set
for name, clf in all_clfs.items():
    start_clf = time.time()
    print('Training classifier: {}. {}'.format(name, clf))
    clf.fit(X_train, y_train)

    print('Score on train set: {}'.format(clf.score(X_train, y_train)))
    print('Score on test set: {}'.format(clf.score(X_test, y_test)))

    y_pred_test = clf.predict(X_test)
    print('Accuracy: {}'.format(accuracy_score(y_test, y_pred_test)))
    print('Precision: {:.3f}'.format(precision_score(y_test, y_pred_test)))
    print('Recall: {:.3f}'.format(recall_score(y_test, y_pred_test)))

    # print(clf.feature_importances_)
    # print(clf.predict(X[0:2, :])    print('Time taken to train and evaluate classifier: {}'.format(time.time() - start_clf))

    if name == 'LogisticRegression':
        # important_features = [unique_words[x] for x in np.argsort(clf.coef_)[::-1][0]][0:50]
        feat_importances_sorted_idx = np.argsort(clf.coef_[0])[::-1]
        important_features = [(unique_words[x], feat_importances[x]) for x in feat_importances_sorted_idx][0:5]
        print(important_features)
    elif name == 'RFC 100 tree':
        # Print the name and gini importance of each feature
        feat_importances = clf.feature_importances_
        feat_importances_sorted_idx = np.argsort(feat_importances)[::-1]
        print([(unique_words[x], feat_importances[x]) for x in feat_importances_sorted_idx][0:50])
        # print([unique_words[x] for x in feat_importances_sorted_idx][0:50])


# Cross-validation 5 way set
# for name, clf in all_clfs.items():
#     start_clf = time.time()
#     print('Training classifier: {}. {}'.format(name, clf))
#     # clf.fit(X_train, y_train)
#     scores = cross_val_score(clf, X, y, cv=5)
#
#     # todo get classifier out
#     print('Score on train set: {}'.format(clf.score(X_train, y_train)))
#     print('Score on test set: {}'.format(clf.score(X_test, y_test)))
#
#     y_pred_test = clf.predict(X_test)
#     print('Accuracy: {}'.format(accuracy_score(y_test, y_pred_test)))
#     print('Precision: {:.3f}'.format(precision_score(y_test, y_pred_test)))
#     print('Recall: {:.3f}'.format(recall_score(y_test, y_pred_test)))
#
#     # print(clf.feature_importances_)
#     # print(clf.predict(X[0:2, :]))
#     print('Time taken to train and evaluate classifier: {}'.format(time.time() - start_clf))

'''
# https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)



'''
