import re


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




