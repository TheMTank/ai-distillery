from aidistillery.data_cleaning import NGramReplacer, clean_raw_text_from_file
from aidistillery.data_cleaning import normalize_text
import os
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

def test_replace_ngram():
    ngr = NGramReplacer()
    sentence = "convolutional neural networks are different from recurrent neural networks"
    sentence_to_get = "convolutional_neural_networks are different from recurrent_neural_networks"
    assert sentence_to_get == ngr.replace_ngram_in_sentence(sentence)


def test_total_cleaner():
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, "text_file_to_clean")
    lines = (clean_raw_text_from_file(path))

    first_line = lines[0]
    second_line = lines[1]

    assert 'test convolutional_neural_networks like machine_learning' == first_line
    assert 'happy latent_dirichlet_allocation everybody' == second_line


def test_normalize_text():
    assert normalize_text("already normalized") == 'already normalized'
    assert normalize_text("This is a fancy title / with subtitle ") == 'this is a fancy title with subtitle'
    assert normalize_text("#@$~(@ $*This has fancy \n symbols in it \n") == 'this has fancy symbols in it'
    assert normalize_text("Oh no a ton of special symbols: $*#@(@()!") == 'oh no a ton of special symbols'
    assert normalize_text("A (2009) +B (2008)") == 'a 2009 b 2008'
    assert normalize_text("1238912839") == '1238912839'
    assert normalize_text("#$@(*$(@#$*(") == ''
    assert normalize_text("Now$ this$ =is= a $*#(ing crazy string !!@)# check") == 'now this is a ing crazy string check'
    assert normalize_text("Also commata, and other punctuation... is not alpha-numeric") == 'also commata and other punctuation is not alphanumeric'
    assert normalize_text(("This goes over\nTwo Lines")) == 'this goes over two lines'
    assert normalize_text('') == ''
