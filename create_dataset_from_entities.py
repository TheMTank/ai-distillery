import json
from pprint import pprint

# fp = '/home/beduffy/all_projects/distiller-AI/data/entities_for_num_files_12.json'
fp = '/home/beduffy/all_projects/distiller-AI/data/entities_for_num_files_1002.json'

with open(fp) as f:
    data = json.load(f)

# pprint(data)
# print('papers: ', data.keys())
print('Num papers: ', len(data.keys()))

for k in data.keys():
    # todo lower case here
    data[k] = [' '.join(ent.strip().replace('\n', ' ').split()) for ent in data[k]]

# remove more than whitespace and newlines
all_entities = [ent for ent_list in data.values() for ent in ent_list]
# print(all_entities)
print('Num entities: ', len(all_entities))

unique_entities = list(set(all_entities))
print('Num unique entities: ', len(unique_entities))
# unique_entities
entity_to_idx = {ent: idx for idx, ent in enumerate(unique_entities)}
idx_to_entity = {idx: ent for idx, ent in enumerate(unique_entities)}

paper_to_idx = {paper: idx for idx, paper in enumerate(list(data.keys()), start=len(unique_entities))}
idx_to_paper = {idx: paper for idx, paper in enumerate(list(data.keys()), start=len(unique_entities))}

all_entities_to_idx = {**entity_to_idx, **paper_to_idx}

'''
For training, datasets contain three files:
train2id.txt: training file, the first line is the number of triples for training. 
Then the following lines are all in the format (e1, e2, rel) which indicates there is a relation rel between e1 and e2 . 
Note that train2id.txt contains ids from entitiy2id.txt and relation2id.txt instead of the names of the entities and relations. If you use your own datasets, please check the format of your training file. Files in the wrong format may cause segmentation fault.

entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.
'''

data_with_ent_indices = {}
for k in data.keys():
    data_with_ent_indices[k] = [entity_to_idx[ent] for ent in data[k]]

# print(data_with_ent_indices['/home/beduffy/all_projects/arxiv-sanity-preserver/data/txt/1709.00849v3.pdf.txt'])

triplets = []
for k in data_with_ent_indices:
    for ent_idx in data_with_ent_indices[k]:
        triplets.append([paper_to_idx[k], ent_idx, 0])

print('Num triplets: ', len(triplets))

with open('data/triplet_openke_dataset/train2id.txt', 'w') as file:
    file.write(str(len(triplets)) + '\n')
    for trp in triplets:
        file.write('{}\t{}\t{}\n'.format(trp[0], trp[1], trp[2]))

with open('data/triplet_openke_dataset/entity2id.txt', 'w') as file:
    file.write(str(len(all_entities_to_idx.keys())) + '\n')
    for k, v in all_entities_to_idx.items():
        # file.write('{}\t{}\t{}'.format(trp[0], trp[1], trp[2]))
        file.write('{}\t{}\n'.format(k, v))

with open('data/triplet_openke_dataset/entity2id.json', 'w') as file:
    json.dump(all_entities_to_idx, file)


