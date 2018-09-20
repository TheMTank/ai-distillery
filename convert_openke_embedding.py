import json
import pickle

import numpy as np

"""
Read output of openke pytorch transe https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/example_train_transe.py
And create format required for fork on word2vec_explorer
"""

input_embedding_fp = 'data/openke_output/embedding_vec_num_epoch_500.json'
with open(input_embedding_fp) as f:
    data = json.load(f)

print(len(data))
print(len(data["ent_embeddings.weight"]))

# df_entity_to_id = pd.read_csv('/home/beduffy/all_projects/distiller-AI/data/triplet_openke_dataset/entity2id.txt', sep='\t', engine='python')#, nrows=1000)
# print(df_entity_to_id.shape)
# print(df_entity_to_id.head(500))


fp = '/home/beduffy/all_projects/distiller-AI/data/triplet_openke_dataset/entity2id.json'
with open(fp) as f:
    entity2id_dict = json.load(f)

embeddings_object = {}
embeddings_object['labels'] = list(entity2id_dict.keys())
embeddings_object['embeddings'] = np.array(data["ent_embeddings.weight"])

output_fp = 'data/openke_output/{}_word2vec_explorer_format.pkl'.format(input_embedding_fp.split('/')[-1].split('.json')[0])
with open(output_fp, 'wb') as f:
    pickle.dump(embeddings_object, f, protocol=pickle.HIGHEST_PROTOCOL)
