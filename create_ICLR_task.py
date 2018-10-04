import sys
import os
import time
import re
from time import mktime
from datetime import datetime
import collections
import pickle
import random
import argparse
# import urllib.request
# import feedparser

from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import numpy as np

import stopwords
# from utils import Config, safe_pickle_dump
db_path = '/home/beduffy/all_projects/arxiv-sanity-preserver/db.p'

# todo upload files to Github?

# lets load the existing database to memory
try:
    print(db_path)
    db = pickle.load(open(db_path, 'rb'))
except Exception as e:
    print('error loading existing database:')
    print(e)
    print('starting from an empty database')
    db = {}

print(len(db.keys()))

# Only take version 1 papers
# all_titles_lowercase = [doc['title'].lower().strip() for k, doc in db.items() if doc['_version'] == 1]
# all_titles_lowercase = [re.sub(' +',' ', doc['title'].lower().strip().replace('\n', '')) for k, doc in db.items()]
all_titles_lowercase = [re.sub(' +',' ', doc['title'].lower().strip().replace('\n', '')) for k, doc in db.items() if doc['_version'] == 1]
# todo try: " ".join(foo.split())
all_dates = [datetime.fromtimestamp(mktime(doc['published_parsed'])) for k, doc in db.items() if doc['_version'] == 1]

print('Num papers counting all versions: {}. Num papers only first version: {}'.format(len(db.keys()), len(all_titles_lowercase)))

# all_titles_words = [word.lower() for title in all_titles for word in title.split()]
# all_titles_words = [word for word in all_titles_words if word not in stopwords.stopwords]
# c = collections.Counter(all_titles_words)

#

with open('data/all_ICLR_submissions/all_ICLR_submissions_titles.txt') as f:
    lines = f.readlines()

    print(len(lines))
    iclr_titles_sub = [iclr_title.strip().lower() for iclr_title in lines]
    direct_matches_arxiv_iclr_sub = [iclr_title for iclr_title in iclr_titles_sub if iclr_title in all_titles_lowercase]

    print('Num ICLR title submissions: ', len(iclr_titles_sub))
    print('First 10 direct matches: ', direct_matches_arxiv_iclr_sub[0:10])
    print('Number of direct matches: ', len(direct_matches_arxiv_iclr_sub))
    # print('First 10 fuzzy matches: ', fuzzy_matches_arxiv_iclr[0:10])
    # print('Number of fuzzy matches: ', len(fuzzy_matches_arxiv_iclr))

# sys.exit()
with open('data/all_ICLR_submissions/ICLR_accepted.txt') as f:
    lines = f.readlines()

    iclr_titles_acc = [iclr_title.strip().lower() for iclr_title in lines]

    print('Progressive Growing of GANs for Improved Quality, Stability, and Variation'.lower() in all_titles_lowercase)
    print('Progressive Growing of GANs for Improved Quality, Stability, and Variation'.lower() in iclr_titles_acc)
    print(iclr_titles_acc[3])
    #[x for x in all_titles_lowercase if 'progressive growing' in x]

    # sys.exit()

    direct_matches_arxiv_iclr_acc = [iclr_title for iclr_title in iclr_titles_acc if iclr_title in all_titles_lowercase]

    matched_submitted_papers_within_accepted = [x for x in direct_matches_arxiv_iclr_sub if x in direct_matches_arxiv_iclr_acc]
    matched_accepted_papers_within_submitted = [x for x in direct_matches_arxiv_iclr_acc if x in direct_matches_arxiv_iclr_sub]
    print(matched_submitted_papers_within_accepted[0:10])
    print(len(matched_submitted_papers_within_accepted))
    print(len(direct_matches_arxiv_iclr_sub))
    print('Number of accepted papers that are within the submitted: {}'.format(len(matched_accepted_papers_within_submitted)))

    print(set(direct_matches_arxiv_iclr_sub) - set(matched_submitted_papers_within_accepted))


    # sys.exit()
    # # fuzzy_matches_arxiv_iclr = [iclr_title for iclr_title in iclr_titles if fuzz.ratio(iclr_title in all_titles_lowercase]
    # fuzzy_matches_arxiv_iclr = []
    # for idx, iclr_title in enumerate(iclr_titles):
    #     print('Idx: {}'.format(idx))
    #     found = False
    #     for arxiv_title in all_titles_lowercase:
    #         if fuzz.ratio(iclr_title, arxiv_title) > 94:
    #             if found:
    #                 print('Found a 2nd or 3rd+ match between:, ', iclr_title, arxiv_title)
    #             fuzzy_matches_arxiv_iclr.append(iclr_title)
    #             found = True

    print('Num ICLR titles: ', len(iclr_titles_acc))
    print('First 10 direct matches: ', direct_matches_arxiv_iclr_acc[0:10])
    print('Number of direct matches: ', len(direct_matches_arxiv_iclr_acc))
    # print('First 10 fuzzy matches: ', fuzzy_matches_arxiv_iclr_acc[0:10])
    # print('Number of fuzzy matches: ', len(fuzzy_matches_arxiv_iclr_acc))

