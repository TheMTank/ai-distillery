import json
import argparse

import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description='')
parser.add_argument('--path-to-timeline', help='path to twitter timeline in json format')

args = parser.parse_args()

with open(args.path_to_timeline, 'r') as f:
    all_tweets = json.load(f)
    print(len(all_tweets))
    # print(all_tweets[1])

    unix_timestamps = [x['unix_timestamp'] for x in all_tweets]

    df = pd.DataFrame({'date': unix_timestamps})

    df['date'] = df['date'].astype(int)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df = df.sort_values(by=['date'])
    print(df.head())

    print(df.describe())

    # generally which months are most popular
    df.groupby(df["date"].dt.month).count().plot(kind="bar")
    plt.show()

    # count over time which just happens to be in order
    df['date'].value_counts().plot()
    plt.show()

    df.groupby(df["date"].dt.day).count().plot(kind="bar")
    plt.show()

    # df['date_day'] = df['date'].apply(lambda x: x.dt.day)
    # counts = df['date_day'].value_counts(sort=False)
    # not working well below. I want a simple line chart.
    counts = df['date'].value_counts(sort=False)
    # plt.bar(counts.index, counts)
    plt.plot(counts.index, counts)
    plt.show()
