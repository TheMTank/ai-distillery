#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Executable to harvest metadata from semanticscholar.org

Takes in a file with one identifier per line:
Each identifier can be either:
- S2PaperID
- arXiv:<arxiv_identifier_without_version>
- doi

Examples for identifiers:
- 0796f6cd7f0403a854d67d525e9b32af3b277331 (S2PaperId)
- 10.1038/nrn3241 (doi)
- arXiv:1705.10311 (ArXivId)

For more information consult https://api.semanticscholar.org

To run the executable, issue:

    python3 harvest_semanticscholar.py id_file.txt >> metadata.jsonl

The file will store its most-recent successfully processed identifier in a file
"harvest_checkpoint.txt" (defined by CHECKPOINT_FILE).  When this file is
present in the current working directory, it is used to restore the checkpoint
and only query semanticscholar for identifiers after the identifier in the
checkpoint file.

"""


import sys
import os
import requests
import json
import time

from tqdm import tqdm

# Checkpoint file
CHECKPOINT_FILE = "harvest_checkpoint.txt"

# Number of retries
RETRIES = 3
# Sleep time between requests
SLEEPTIME = 1
# Sleep time between retries (good if longer than default)
RETRY_SLEEPTIME = 5

def ask_semanticscholar(identifier):
    r = requests.get('https://api.semanticscholar.org/v1/paper/' + identifier)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()



def main():
    import fileinput
    if os.path.isfile(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as chk_file:
            checkpoint = chk_file.read().strip()
            print("Using checkpoint: ", checkpoint, file=sys.stderr)
    else:
        checkpoint = None

    for line in tqdm(fileinput.input()):
        id = line.strip()
        if checkpoint is not None:
            if id == checkpoint:
                print("Checkpoint ", checkpoint, "restored.", file=sys.stderr)
                checkpoint = None
                continue
            else:
                continue

        for __i in range(RETRIES):
            try:
                data = ask_semanticscholar(id)
                print(data)
                break
            except Exception:
                time.sleep(RETRY_SLEEPTIME)
                continue
        else:
            raise UserWarning("Number of retries exceeded on id: " + id + " Please retry later")

        with open("harvest_checkpoint.txt", 'w') as chk_file:
            print(id, file=chk_file)
        time.sleep(SLEEPTIME)

if __name__ == "__main__":
    main()
