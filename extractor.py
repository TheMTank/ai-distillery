import argparse
import os
import glob
try:
    import spacy
    nlp = spacy.load('en')
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

# This is just a heuristic as most interesting stuff is labeled as 'ORG': Wasserstein GAN, AI, ML, and more...
# You can supply custom set of entities via command line args `-e ORG PERSON QUANTITY`
DEFAULT_ENTITY_TYPES = { 'ORG' } 

def extract_entities(document, types):
    """
    Arguments
    ---------
    documents: an iterable of strings
    types: entitytypes to extract 

    Returns
    -------
    List of strings representing the entities (as of now, Labels are discarded)
    """
    return [ent.text for ent in nlp(document).ents if ent.label_ in types]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default="data/txt",
                        help="Root directory for txt files [data/txt]")
    parser.add_argument('-e', '--entity-types', nargs='+',
                        default=DEFAULT_ENTITY_TYPES, choices=VALID_ENTITY_TYPES,
                        help="Specify entity types to extract. Default: {}".format(DEFAULT_ENTITY_TYPES))

    args = parser.parse_args()
    # set for faster access
    entity_types = set(args.entity_types)
    print("Extracting entities of types:", entity_types, "from '*.txt' documents in", args.data)

    # Generator
    paths = glob.iglob(os.path.join(args.data, '*.txt'))

    for path in paths:
        with open(path, 'r') as fhandle:
            print("Processing:", fhandle.name)
            doc = fhandle.read()
            ents = extract_entities(doc, entity_types)
            print("{}: {}".format(fhandle.name, ents))


if __name__ == '__main__':
    main()



