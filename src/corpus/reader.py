from collections import Counter
import os

from ..corpus.corpus_data import CorpusData
from ..helper import functional as fct
from ..helper import paths_generator as pgen
from ..helper import utils as utl


class Reader:
    def __init__(self, args):
        import src
        def_books_paths = os.listdir(os.path.join(pgen.parent_path(src.SRC_DIR), "data/kidsbooks"))

        self.paths_books = args.paths_books if args.paths_books else def_books_paths
        self.ctx = args.ngrams
        self.read_corpus = args.read_corpus
        self.save_corpus = args.save_corpus

    @staticmethod
    def line_rule(line):
        ln = len(line) if len(line) else -1
        return ln > 5 and len([xi for xi in line if xi.isalpha()]) / ln >= 0.6

    def generate_pad(self, n):
        return ["<PAD>"] * (self.ctx - n)

    def generate_pads(self, left, right):
        return self.generate_pad(len(left)) + left, right + self.generate_pad(len(right))

    def generate_left_idx(self, i):
        return [x for x in range(i - self.ctx, i) if x >= 0]

    def generate_right_idx(self, i, ln):
        return [x for x in range(i + 1, i + 1 + self.ctx) if x < ln]

    def generate_idx(self, i, s):
        return [s[i] for i in self.generate_left_idx(i)], [s[i] for i in self.generate_right_idx(i, len(s))]

    def extract_context(self, i, sentence):
        return self.generate_pads(*self.generate_idx(i, sentence))

    # TODO: (just as an example); check what will be needed for the generation
    def generate_ngrams(self, sentence):
        res = []
        for i, target in enumerate(sentence):
            left_ctx, right_ctx = self.extract_context(i, sentence)
            res.append((target, left_ctx + right_ctx))

        return res

    # TODO: definitely needs a refactor and optimization work
    def generate_windowed_sentences(self, sentences, word2idx, window):
        res = []
        for sen in sentences:
            # tok = [word2idx[x] for s in sen for x in s.split()]
            tok = [word2idx[s] for s in sen.split()]
            if len(tok) < window:
                ile = window - len(tok)
                res.append(tok + [word2idx["<PAD>"]] * ile)
            elif len(tok) > window:
                ile = len(tok) - window
                for i in range(0, ile):
                    res.append(tok[i:i + window])
            else:
                res.append(tok)

        return res

    @staticmethod
    def fil(word):
        return ''.join([x for x in word if x.isalpha() or x.isnumeric() or x in "!?,.-:;\"'"])

    @staticmethod
    def idexify(ngrams, word2idx):
        return [(word2idx[t], [word2idx[ci] for ci in c]) for t, c in ngrams]

    def preprocess(self):
        # ngrams = []

        # TODO: check whether distinguishing one book from the other matters
        #       in terms of text generation (same with chapters, etc.)

        for book in fct.flatten(pgen.generate_absolute_paths(self.paths_books)):
            with open(book) as w:
                # TODO: split on end of sentence punctuation marks (dot is not always end of sentence);
                #       think about punctuation in general

                lines = [line.strip()
                             .replace("\n", "")
                             .replace("'t", " not")
                             .replace("'s", " is")
                             .replace("'d", "would")
                             .replace("'ve", "have")
                         for line in w.read().split("\n\n") if self.line_rule(line)]
                tokenized_sentences = [' '.join([self.fil(word.lower()) for word in sentence.split()]) for sentence in ' '.join(lines).split(".")]
                tokenized_sentences = [sen for sen in tokenized_sentences if sen]

                naive_tokens = [token for sen in tokenized_sentences for token in sen.split()]

                word2freq = Counter(naive_tokens)
                vocabulary = list(word2freq.keys())
                word2idx = {"<PAD>": 0}
                word2idx.update({token: i + 1 for i, token in enumerate(vocabulary)})
                idx2word = {v: k for k, v in word2idx.items()}

                window = 4
                # decided on windowed sentences
                windowed_sentences = self.generate_windowed_sentences(tokenized_sentences, word2idx, window)

                # TODO: consider removing too seldom target words and their context
                # ngrams.append([self.generate_ngrams(sentence) for sentence in naive_tokens if len(sentence)])

            # assert len(ngrams) == len(self.paths_books)

            # return self.idexify(fct.flatten(fct.flatten(ngrams)), word2idx), vocabulary, word2idx, idx2word
            return windowed_sentences, vocabulary, word2idx, idx2word

    def read(self):
        if self.read_corpus and len(os.listdir('/'.join(self.read_corpus.split("/")[:-1]) + "/")):
            corpus_data = utl.load_obj(self.read_corpus)

            ln = len(corpus_data.word2idx)
            assert ln > 100, (f"Corpus is either too small (size: {ln}), or it has not been created yet; "
                              f"either way - standard procedure will continue")
        else:
            corpus_data = CorpusData(*self.preprocess())

            if self.save_corpus:
                utl.save_obj(corpus_data, self.save_corpus)

        return corpus_data
