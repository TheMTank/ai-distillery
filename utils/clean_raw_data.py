import os
import re
try: 
    from nltk.corpus import stopwords
    # from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

def remove_stop_words(string):
    if not NLTK_AVAILABLE:
        raise UserWarning("Please install nltk to make use of stop_words")

    stop_words = set(stopwords.words('english'))

    words = string.split()


    unstopped = [w for w in words if not w in stop_words]

    return " ".join(unstopped)

def filter_empty(string):
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
    print("Text File Loaded")
    print("Now Cleaning")
    content = [x.strip().replace("\n", "").lower() for x in content]
    content = filter(lambda x: len(x) > min_length, content)  # filter min length
    content = [s.split() for s in content]
    content = [map(lambda x : remove_non_alpha_chars(x), s) for s in content]
    content = [filter(lambda  x : x != "", s) for s in content]

    return content



# An alternative short hand
def normalize_text(text):
    """
    Normalizes a string.
    The string is lowercased and all non-alphanumeric characters are removed.

    >>> normalize("already normalized")
    'already normalized'
    >>> normalize("This is a fancy title / with subtitle ")
    'this is a fancy title with subtitle'
    >>> normalize("#@$~(@ $*This has fancy \\n symbols in it \\n")
    'this has fancy symbols in it'
    >>> normalize("Oh no a ton of special symbols: $*#@(@()!")
    'oh no a ton of special symbols'
    >>> normalize("A (2009) +B (2008)")
    'a 2009 b 2008'
    >>> normalize("1238912839")
    '1238912839'
    >>> normalize("#$@(*$(@#$*(")
    ''
    >>> normalize("Now$ this$ =is= a $*#(ing crazy string !!@)# check")
    'now this is a ing crazy string check'
    >>> normalize("Also commata, and other punctuation... is not alpha-numeric")
    'also commata and other punctuation is not alphanumeric'
    >>> normalize(("This goes over\\n" "Two Lines"))
    'this goes over two lines'
    >>> normalize('')
    ''
    """
    return ' '.join(filter(None, (''.join(c for c in w if c.isalnum())
                                  for w in text.lower().split())))


##############################################################################
# The following method is technically no data cleaning but resolving filenames
# to ids Shift this method somewhere else if appropraite
# Used by: ../lsa.py

def identifier_from_path(path):
    """
    Gets the arxiv identifier from a path

    >>> identifier_from_path("0507037v3.pdf.txt")
    '0507037v3'
    >>> identifier_from_path("0705.4676v8.pdf.txt")
    '0705.4676v8'
    >>> identifier_from_path("data/txt/0705.4676v8.pdf.txt")
    '0705.4676v8'
    """
    basename = os.path.basename(path)
    return os.path.splitext(os.path.splitext(basename)[0])[0]
