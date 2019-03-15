import argparse
import re
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# import stopwords
parser = argparse.ArgumentParser(description='')
parser.add_argument('--db-path', help='')

args = parser.parse_args()

try:
    print(args.db_path)
    db = pickle.load(open(args.db_path, 'rb'))
except Exception as e:
    print('error loading existing database:')
    print(e)
    print('starting from an empty database')
    db = {}

print(len(db.keys()))

preprocessed_db = [{'title': re.sub(' +', ' ', doc['title'].lower().strip().replace('\n', '')),
                    'summary': doc['summary']} for k, doc in db.items()]
surveys = [x['summary'] for x in preprocessed_db if 'a survey' in x['title']]
not_surveys = [x['summary'] for x in preprocessed_db if 'a survey' not in x['title']]
not_surveys_titles = [x['title'] for x in preprocessed_db if 'a survey' not in x['title']]

vectorizer = TfidfVectorizer()

X = surveys + not_surveys[:len(surveys)]
y = np.array([1] * len(surveys) + [0] * len(surveys))

X = vectorizer.fit_transform(X)
vectoriser_feat_names = vectorizer.get_feature_names()
print(vectoriser_feat_names[0:20])
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
print(clf.feature_importances_)

X_not_surveys = vectorizer.transform(not_surveys)
X_proba = clf.predict_proba(X_not_surveys)

top_indices = np.argsort(X_proba[:, 1])[::-1]
print([not_surveys_titles[idx] for idx in top_indices[:20]])

top_feat_indices = np.argsort(clf.feature_importances_)[::-1]
print([vectoriser_feat_names[idx] for idx in top_feat_indices[0:50]])
