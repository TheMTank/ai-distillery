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

# todo what paper what is march 2014 if the original came out in june??? Check for words with letters "gan" consecutively
# todo sort by time
# todo do word count as well

# lets load the existing database to memory
try:
    print(db_path)
    db = pickle.load(open(db_path, 'rb'))
except Exception as e:
    print('error loading existing database:')
    print(e)
    print('starting from an empty database')
    db = {}

all_gan_titles_date = [[doc['title'], datetime.fromtimestamp(mktime(doc['published_parsed']))] for k, doc in db.items() if doc['_version'] == 1 and 'gan' in doc['title'].lower()]
# all_dates = [datetime.fromtimestamp(mktime(doc['published_parsed'])) for k, doc in db.items() if doc['_version'] == 1]
all_dates = [x[1] for x in all_gan_titles_date]

all_gan_titles_date.sort(key=lambda x: x[1])#, reverse=True)
# todo remove organic, oganising, organizing, more

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
plt.title('Num papers with GANs in their title released on arxiv over time up to end of September (cs.[CV|CL|LG|AI|NE] / stat.ML)')
plt.grid(True, color='darkgray', alpha=0.6)
# plt.grid(b=True, which='major', color='b', linestyle='-')
# plt.xticks(rotation=80)
# plt.savefig("test.png")

print([list(x) for x in list(zip(list(x_ticks), list(num_in_each_year_month)))])

plt.show()
print(num_in_each_year_month)
print(list(x_ticks))

