import json
from pprint import pprint

with open('data/semanticscholar/1412.6980_adam_sample.json') as f:
    data = json.load(f)

pprint(data)

print()

list_of_topics = [x['topic'] for x in data['topics']]
print(data['topics'])
print(list_of_topics)

print('Number of citations: {}'.format(len(data['citations'])))

print(data.keys())
