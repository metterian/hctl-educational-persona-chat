import os
import spacy

nlp = spacy.load(
    "en_core_web_sm",  # set the tokenizer
)


def get_paradir_path(path: str, relative=True) -> str:
    """
    Get the file path including parents path
        - relative : for debugging
    """
    if relative:
        return os.path.join(os.path.pardir, path)
    else:
        return path


def check_punctuation(words):
    punctuation = [".", ","]
    return [word for word in words if word in punctuation]


# preprocess the text
def space_before_eos(sentence: str, tokenizer=nlp):
    """
    Preprocessing function:
    - Spaces before periods at end of sentences
    - everything lowercase
    """
    table = str.maketrans({".": " .", ",": " ,"})
    sentence = sentence.lower().split()
    for i, word in enumerate(sentence):
        if check_punctuation(word):
            for token in tokenizer(word):
                if token.pos_ == "PUNCT":
                    sentence[i] = sentence[i].translate(table)
    return " ".join(sentence)
