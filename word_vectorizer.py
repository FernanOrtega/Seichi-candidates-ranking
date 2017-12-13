import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim, os, time
from word_preprocessing import Preprocessor


class WordEmbeddings(object):
    def __init__(self, lang, modelpath):
        self.preprocessor = Preprocessor(lang)
        if os.path.isfile(modelpath):
            self.model = gensim.models.Word2Vec.load(modelpath)
        else:
            raise FileNotFoundError

    def word2idx(self, word, do_prep=True):
        if do_prep:
            p_word = self.preprocessor.preprocess(word)
        else:
            p_word = word
        return self.model.wv.vocab.get(p_word, self.model.wv.vocab.get('UNK')).index

    def idx2word(self, idx):
        return self.model.wv.index2word[idx]
