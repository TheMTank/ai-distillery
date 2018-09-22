"""
ner.py (Executable)

Extract named entities from a text collection

Example to extract all entitys of type ORG: `python3 ner.py -d path/to/data -e ORG > orgs.yaml`
"""
import argparse
import os
import glob
import json

try:
    import spacy
    NLP = spacy.load('en')
except ImportError:
    raise UserWarning("Please install python package spacy")
except OSError:
    raise UserWarning("Please install the English model from spacy: `python -m spacy download en`")

# Complete list extracted from https://spacy.io/usage/linguistic-features#entity-types
VALID_ENTITY_TYPES = {
    "PERSON",                         # People, including fictional.
    "NORP",                           # Nationalities or religious or political groups.
    "FAC",                            # Buildings, airports, highways, bridges, etc.
    "ORG",                            # Companies, agencies, institutions, etc.
    "GPE",                            # Countries, cities, states.
    "LOC",                            # Non-GPE locations, mountain ranges, bodies of water.
    "PRODUCT",                        # Objects, vehicles, foods, etc. (Not services.)
    "EVENT",                          # Named hurricanes, battles, wars, sports events, etc.
    "WORK_OF_ART",                    #	Titles of books, songs, etc.
    "LAW",                            # Named documents made into laws.
    "LANGUAGE",                       #	Any named language.
    "DATE",                           # Absolute or relative dates or periods.
    "TIME",                           # Times smaller than a day.
    "PERCENT",                        # Percentage, including "%".
    "MONEY",                          # Monetary values, including unit.
    "QUANTITY",                       #	Measurements, as of weight or distance.
    "ORDINAL",                        # "first", "second", etc.
    "CARDINAL"                        #	Numerals that do not fall under another type.
}

# This is just a heuristic as most interesting stuff is labeled as 'ORG':
# Wasserstein GAN, AI, ML, and more...  You can supply custom set of entities
# via command line args `-e ORG PERSON QUANTITY`
DEFAULT_ENTITY_TYPES = {'ORG'}

def extract_entities(document, types):
    """
    Arguments
        ---------
        documents: an iterable of strings
        types: entity types to extract

        Returns
        -------
        List of strings representing the entities (as of now, Labels are discarded)
    """
    return [ent.text for ent in NLP(document).ents if ent.label_ in types]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default="data/txt",
                        help="Root directory for txt files [data/txt]")
    parser.add_argument('-e', '--entity-types', nargs='+',
                        default=DEFAULT_ENTITY_TYPES, choices=list(VALID_ENTITY_TYPES) + ['_all'],
                        help=("Specify entity types to extract."
                              "Default: {}").format(DEFAULT_ENTITY_TYPES))
    parser.add_argument('--num-files', default=1000, type=int,
                        help="Choose the number of files to process since execution can take hours")
    parser.add_argument('-o', '--output-path', default='data/entities_for',
                        help="Choose where to output json file containing all entities with number of files added on")

    args = parser.parse_args()
    # Resolve keyword '_all' as alias for all entity types
    entity_types = VALID_ENTITY_TYPES if '_all' in args.entity_types else args.entity_types
    # set for faster access
    entity_types = set(entity_types)
    print("Extracting entities of types:", entity_types, "from '*.txt' documents in", args.data)

    file_paths = glob.glob(os.path.join(args.data, '*.txt'))
    num_files = len(file_paths)
    print('Num files at path {}: {}'.format(args.data, num_files))
    entities_for_file = {}
    for idx, path in enumerate(file_paths):
        with open(path, 'r') as fhandle:
            try:
                print("Processing:", fhandle.name)
                doc = fhandle.read()
                ents = extract_entities(doc, entity_types)
                # print("{}: {}".format(fhandle.name, ents))
                entities_for_file[fhandle.name] = ents
            except ValueError as e:
                print('Error reading file: \n {}'.format(e))

        if idx % 100 == 0:
            print('{}/{}'.format(idx, num_files))

        # Execution takes a long time so break after num_files have been reached
        if idx > args.num_files:
            break

    with open('{}_num_files_{}.json'.format(args.output_path, len(entities_for_file.keys())), 'w') as fp:
        json.dump(entities_for_file, fp)

if __name__ == '__main__':
    main()
