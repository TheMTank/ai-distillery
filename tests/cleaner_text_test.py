from aidistillery.data_cleaning import NGramReplacer, clean_raw_text_from_file
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
