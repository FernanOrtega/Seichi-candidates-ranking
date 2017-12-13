import re
from nltk import SnowballStemmer

map_langs = {'es': 'spanish', 'en': 'english', 'fr': 'french', 'it': 'italian', 'pt': 'portuguese'}


class Preprocessor(object):
    def __init__(self, lang):
        s_lang = map_langs.get(lang, lang)
        self.re_digits = re.compile(r"^[0-9]+(?:[,.][0-9]+)*[ºªkKmM]?$")
        self.re_mail = re.compile(r"^[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+$")
        self.re_url = re.compile(
            r"^[A-Za-z0-9-_]+:\/\/[A-Za-z0-9-_]*(?:\.[A-Za-z0-9-_]+)*|[A-Za-z0-9-_]+(?:\.[A-Za-z0-9-_]+)+$")
        if SnowballStemmer.languages.__contains__(s_lang):
            self.stemmer = SnowballStemmer(s_lang)
        else:
            self.stemmer = SnowballStemmer('porter')
        self.mapstem = {}

    def preprocess(self, word):
        if self.re_digits.match(word):
            p_word = 'DIGITS'
        elif self.re_mail.match(word):
            p_word = 'EMAIL'
        elif self.re_url.match(word):
            p_word = 'URL'
        else:
            l_word = word.lower()
            p_word = self.mapstem.get(l_word)
            if p_word is None:
                p_word = self.stemmer.stem(word.lower())
                self.mapstem.update({l_word: p_word})
        return p_word
