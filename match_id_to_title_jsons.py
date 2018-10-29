import pandas as pd
import json

with open('data/all_ICLR_submissions/all_ICLR_accepted_id_to_title.json') as f:
    all_ICLR_accepted_id_to_title = json.load(f)

with open('data/all_ICLR_submissions/all_ICLR_submission_id_to_title.json') as f:
    all_ICLR_submission_id_to_title = json.load(f)

print('Num accepted ids: {}. Num submitted ids: {}'.format(len(all_ICLR_accepted_id_to_title), len(all_ICLR_submission_id_to_title)))

accepted_keys, submitted_keys = list(all_ICLR_accepted_id_to_title.keys()), list(all_ICLR_submission_id_to_title.keys())
# unique_accepted, unique_submitted = set(all_ICLR_accepted_id_to_title.keys()), set(all_ICLR_submission_id_to_title.keys())
unique_accepted, unique_submitted = set(accepted_keys), set(submitted_keys)
print('Num unique accepted ids: {}. Num unique submitted ids: {}'.format(len(unique_accepted), len(unique_submitted)))

rejected_or_withdrawn = unique_submitted - unique_accepted
accepted_not_in_submitted = unique_accepted - unique_submitted
accepted_within_submitted_direct_match = [x for x in unique_accepted if x in unique_submitted]
# accepted_without_ids_submitted_doesnt_have = [x for x in accepted_within_submitted_direct_match if x in ]

print(rejected_or_withdrawn)
print(accepted_not_in_submitted)
print('Num rejected_or_withdrawn: {}. Num accepted_not_in_submitted (with direct match): {}'.format(len(rejected_or_withdrawn), len(accepted_not_in_submitted)))
print('Num accepted_within_submitted: {}'.format(len(accepted_within_submitted_direct_match)))
# print('Num accepted_without_ids_submitted_doesnt_have: {}'.format(len(accepted_without_ids_submitted_doesnt_have)))

# all_ICLR_accepted_id_to_title_matched = {}

id_title_accepted = []
for paper_id in unique_submitted:
    id_title_accepted.append({'paper_id': paper_id, 'title': all_ICLR_submission_id_to_title[paper_id],
                              'accepted': 1 if paper_id in accepted_within_submitted_direct_match else 0})

df_id_title_accepted = pd.DataFrame(id_title_accepted)
df_id_title_accepted = df_id_title_accepted[['paper_id', 'title', 'accepted']]
print(df_id_title_accepted.head())
print(df_id_title_accepted.columns)

df_id_title_accepted.to_csv('data/all_ICLR_submissions/ICLR_binary_classification.csv')
