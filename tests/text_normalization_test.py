from aidistillery.data_cleaning import normalize_text

def test_text_normalization():
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
