import os
import time
from time import mktime
from datetime import datetime
import collections
import pickle
import random
import argparse
# import urllib.request
# import feedparser

import stopwords

import matplotlib.pyplot as plt
import numpy as np

# from utils import Config, safe_pickle_dump

db_path = '/home/beduffy/all_projects/arxiv-sanity-preserver/db.p'

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

# todo Get all RL paper titles from all time
# todo Simple tokenized keyword counting (fix problems) on all titles+abstracts
# todo Store all papers+titles in a cool JSON data structure with extra info
# todo Finish keyword counting with n-grams
# todo Links and even web page or slack bots to display everything easily
# todo group each paper into an application area.
# todo Create Visualisations and stats. Bokeh, plotly or some other python tool? Show on a web page
# todo overlay exponential chart over number of papers chart
# todo start putting labels on each paper
# todo create RNN to generate paper
# todo create RNN for language modelling
# todo cluster documents
# todo tfidf on all docs and n-grams
# todo doc2vec. word2vec. sentence2vec
# todo search and information retrieval and
# todo look at karpathy's html
# todo graph all authors and graph all papers. find edges and links and who writes with who. Find institutions. Create google scholar
# todo get all papers back to start of 2014, download them all, get text for them all
# todo get mentions of all frameworks (TF, pytorch, etc)
# todo start categorising them all automatically
# todo start plotting the interest in certain techniques e.g. when was height of GAN fever? How many RL papers each month?
# todo stacked line chart showing what each paper is categorised into
# todo topic modelling and then more complex Topic Modelling
# todo abstracts, titles and whole texts and their weight
# todo stemming? nltk tokenize? Remove words mentioned less than 3 times.
# todo automatic popularity testing. With twitter or through citations (Google Scholar)
# todo calculate how much rise each year compared to last
# todo Automate everything
# todo become knowledgeable about everything in AI and have the authority and stats to back it up because i have 40k papers sitting on my harddrive
# todo Write about it or list it somewhere public

# Only take version 1 papers
all_titles = [doc['title'] for k, doc in db.items() if doc['_version'] == 1]
all_dates = [datetime.fromtimestamp(mktime(doc['published_parsed'])) for k, doc in db.items() if doc['_version'] == 1]

print('Num papers counting all versions: {}. Num papers only first version: {}'.format(len(db.keys()), len(all_titles)))

all_titles_words = [word.lower() for title in all_titles for word in title.split()]
all_titles_words = [word for word in all_titles_words if word not in stopwords.stopwords]
c = collections.Counter(all_titles_words)

for t in c.most_common(200):
    print(t)

# Collect how many papers in each month and in each year
dt_year = collections.defaultdict(list)
dt_month_in_year = collections.defaultdict(list)
for dt in all_dates:
    dt_year[str(dt.year)].append(dt)
    dt_month_in_year[str(dt.year) + '-' + str(dt.month)].append(dt)

# Create lists of how many papers in each month sequentially
x_ticks = []
num_in_each_year_month = []
for year in ['2014', '2015', '2016', '2017', '2018']:
    for month in range(1, 13):
        if year == '2018' and month == 10:
            break
        key = str(year) + '-' + str(month)
        num_in_year_month = len(dt_month_in_year[key])
        print(key + ': ' + str(num_in_year_month))
        num_in_each_year_month.append(num_in_year_month)
        x_ticks.append(key[2:])

    print('\nTotal ' + year + ': ' + str(len(dt_year[year])), '\n')

# Graph it
# plt.style.use('seaborn') # ggplot
plt.plot(range(len(num_in_each_year_month)), num_in_each_year_month)
plt.xticks(range(len(num_in_each_year_month)), x_ticks, rotation='vertical')
# plt.xticks(range(len(num_in_each_year_month)), x_ticks)
a = [x.set_color("red") for idx, x in enumerate(plt.gca().get_xticklabels()) if (idx) % 12 == 0]
# a = [x.set_visible(False) for idx, x in enumerate(plt.gca().get_xticklabels()) if (idx) % 12 != 0]
# a = [x.set_majorformatter(3) for idx, x in enumerate(plt.gca().get_xticklabels()) if (idx) % 12 != 0]

plt.tick_params()
plt.xlabel('Month')
plt.ylabel('Number of papers')
plt.title('Num papers released on arxiv over time up to end of June (cs.[CV|CL|LG|AI|NE] / stat.ML)')
plt.grid(True, color='darkgray', alpha=0.6)
# plt.grid(b=True, which='major', color='b', linestyle='-')
# plt.xticks(rotation=80)
# plt.savefig("test.png")

print([list(x) for x in list(zip(list(x_ticks), list(num_in_each_year_month)))])

plt.show()
print(num_in_each_year_month)
print(list(x_ticks))
