import pandas as pd

import glob

all_text_files = glob.glob('../search_iclr_2018/text/papers/*')

print(len(all_text_files))
print(all_text_files[0:5])


df_ICLR_old_binary = pd.read_csv('data/all_ICLR_submissions/ICLR_binary_classification.csv')

ICLR_old_ids = set(df_ICLR_old_binary['paper_id'])
print(len(ICLR_old_ids))

c = 0
for text_file in all_text_files:
    if text_file.split('id=')[-1] in ICLR_old_ids:
        c += 1
    else:
        print(text_file.split('id=')[-1])

print(c)
