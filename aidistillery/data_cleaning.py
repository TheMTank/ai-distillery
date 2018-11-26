import re
import gensim
from gensim.models.phrases import Phrases, Phraser
import os
import pandas as pd

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

# try:
#     from nltk.corpus import stopwords
#     # from nltk.tokenize import word_tokenize
#     NLTK_AVAILABLE = True
# except ImportError:
#     NLTK_AVAILABLE = False

def remove_stop_words(string):
    # old variant:
    # stop_words = set(stopwords.words('english'))
    stop_words = ENGLISH_STOP_WORDS

    words = string.split()

    unstopped = [w for w in words if not w in stop_words]

    return " ".join(unstopped)

def filter_empty(string):
    """
    Removes unused spaces from string

    >>> filter_empty("data a  a   a ")
    'data a a a'
    """

    content = string.split()
    content = [filter(lambda  x : x != "", s) for s in content]

    return " ".join(content)

def remove_non_alpha_chars(word):
    word = re.sub("\S*\d\S*", "", word).strip()
    word = re.sub('[^A-Za-z]', '', word).strip()

    return word

def clean_raw_text_from_file(file_name, min_length=0):
    with open(file_name) as f:
        content = f.readlines()
    ngr = NGramReplacer()

    content = map(lambda x : normalize_text(x), content)
    content = map(lambda x : remove_stop_words(x), content)
    content = map(lambda x : ngr.replace_ngram_in_sentence(x), content)
    content = filter(lambda x: len(x) > min_length, content)

    return list(content)

def list_of_strings_to_list_of_lists(content):
    return [s.split() for s in content]


def phrasing_sentences(sentences):
    phrases_bi = Phrases(sentences, min_count=5, threshold=1)
    bigram = Phraser(phrases_bi)
    sentences = map(lambda x: x, bigram[sentences])
    phrases_tri = Phrases(sentences, min_count=5, threshold=1)
    trigram = Phraser(phrases_tri)
    return map(lambda x: x, trigram[sentences])

class NGramReplacer:

    def __init__(self):
        script_dir = os.path.dirname(__file__)
        path = os.path.join(script_dir, "n-grams.txt")
        data = pd.read_csv(path, names=["index", "value"])
        zipped_data = zip(data["index"].values, data["value"].values)
        self.ngrams = dict((y, x) for x, y in zipped_data)

    def replace_ngrams_in_content(self, content):
        content = map(lambda x : self.replace_ngram_in_sentence(x), content)
        return content

    def replace_ngram_in_sentence(self, sentence):
        for key in self.ngrams.keys():
            sentence = sentence.replace(self.ngrams[key].strip(),key)
        return sentence

# An alternative short hand
def normalize_text(text):
    """
    Normalizes a string.
    The string is lowercased and all non-alphanumeric characters are removed.

    >>> normalize_text("already normalized")
    'already normalized'
    >>> normalize_text("This is a fancy title / with subtitle ")
    'this is a fancy title with subtitle'
    >>> normalize_text("#@$~(@ $*This has fancy \\n symbols in it \\n")
    'this has fancy symbols in it'
    >>> normalize_text("Oh no a ton of special symbols: $*#@(@()!")
    'oh no a ton of special symbols'
    >>> normalize_text("A (2009) +B (2008)")
    'a 2009 b 2008'
    >>> normalize_text("1238912839")
    '1238912839'
    >>> normalize_text("#$@(*$(@#$*(")
    ''
    >>> normalize_text("Now$ this$ =is= a $*#(ing crazy string !!@)# check")
    'now this is a ing crazy string check'
    >>> normalize_text("Also commata, and other punctuation... is not alpha-numeric")
    'also commata and other punctuation is not alphanumeric'
    >>> normalize_text(("This goes over\\n" "Two Lines"))
    'this goes over two lines'
    >>> normalize_text('')
    ''
    """
    return ' '.join(filter(None, (''.join(c for c in w if c.isalnum())
                                  for w in text.lower().split())))




