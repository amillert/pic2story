from collections import Counter
import os

from ..corpus.corpus_data import CorpusData
from ..helper import paths_generator as pgen
from ..helper import utils as utl


class Reader:
    def __init__(self, args):
        self.paths_books = args.paths_books
        self.ctx = args.ngrams
        self.read_corpus = args.read_corpus
        self.save_corpus = args.save_corpus
        self.books_to_read = args.books if args.books else 500

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
    @staticmethod
    def generate_windowed_sentences(sentences, word2idx, window):
        res = []
        for sen in sentences:
            tokens = [word2idx[s] for s in sen.split()]
            abs_diff = abs(window - len(tokens))

            if len(tokens) < window:
                res.append(tokens + [word2idx["<PAD>"]] * abs_diff)
            else:
                for i in range(0, abs_diff + 1):
                    res.append(tokens[i:i + window])

        return res

    @staticmethod
    def fil(word):
        return ''.join([x for x in word if x.isalpha() or x.isnumeric() or x in "!?,.-:;\"'"])

    def preprocess(self):
        # TODO: check whether distinguishing one book from the other matters
        #       in terms of text generation (same with chapters, etc.)

        books = []
        for book in pgen.generate_absolute_paths(self.paths_books)[:self.books_to_read]:
            with open(book) as w:
                # TODO: split on end of sentence punctuation marks (dot is not always end of sentence);
                #       think about punctuation in general
                try:
                    lines = [line.strip()
                                 .replace("\n", "")
                                 .replace("'t", " not")
                                 .replace("'s", " is")
                                 .replace("'d", " would")
                                 .replace("'ve", "have")
                             for line in w.read().split("\n\n") if self.line_rule(line)]
                    books.extend(lines)
                except:
                    pass

        tokenized_sentences = [' '.join([self.fil(word.lower()) for word in sentence.split()])
                               for sentence in ' '.join(books).split(".")]
        tokenized_sentences = [sen for sen in tokenized_sentences if sen]
        print("tokenization done")

        naive_tokens = [token for sen in tokenized_sentences for token in sen.split()]

        word2freq = Counter(naive_tokens)
        print("word2freq done")
        vocabulary = ["<PAD>"] + list(word2freq.keys())
        print("vocab done")
        word2idx = {"<PAD>": 0}
        word2idx.update({token: i + 1 for i, token in enumerate(vocabulary)})
        print("word2idx done")
        idx2word = {v: k for k, v in word2idx.items()}
        print("index done")

        window = 4
        # decided on windowed sentences
        windowed_sentences = self.generate_windowed_sentences(tokenized_sentences, word2idx, window)
        print("windowing done")

        return windowed_sentences, vocabulary, word2idx, idx2word

    def read(self):
        contents = os.listdir('/'.join(self.read_corpus.split("/")[:-1]) + "/")
        if self.read_corpus and len(contents) and self.read_corpus.split("/")[-1] in contents:
            corpus_data = utl.load_obj(self.read_corpus)
        else:
            corpus_data = CorpusData(*self.preprocess())

            if self.save_corpus:
                utl.save_obj(corpus_data, self.save_corpus)

        return corpus_data
