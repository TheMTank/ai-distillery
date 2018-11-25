import time
import sys
import requests

BASE_URL_PDF = 'https://openreview.net/pdf?id='

with open('data/all_ICLR_submissions/all_ICLR_submission_IDs.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]

    urls_on_submission_page = ['{}{}'.format(BASE_URL_PDF, line) for line in lines]

num_url_submissions = len(urls_on_submission_page)

for idx in range(1, 1000):
    url = urls_on_submission_page[idx]
    r = requests.get(url, allow_redirects=True)
    save_path = 'data/all_ICLR_submissions/pdfs/{}.pdf'.format(url.split('/')[-1])
    open(save_path, 'wb').write(r.content)
    print('Idx: {}/{}. Saved file to path: {}'.format(idx + 1, num_url_submissions, save_path))

    # be nice
    time.sleep(7)
