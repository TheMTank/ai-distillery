#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import os
import requests
import json
import time

from tqdm import tqdm

CHECKPOINT_FILE = "harvest_checkpoint.txt"

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
            
        data = ask_semanticscholar(id)
        print(data)

        with open("harvest_checkpoint.txt", 'w') as chk_file:
            print(id, file=chk_file)
        time.sleep(1)






if __name__ == "__main__":
    main()
