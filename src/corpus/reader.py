from collections import Counter
import os
import pickle

from src.corpus.preprocess.corpus_data import CorpusData
from src.helper import paths_generator as pgen


class Reader:
    def __init__(self, args):
        self.paths_books = args.paths_books
        self.read_corpus = args.read_corpus
        self.save_corpus = args.save_corpus
        self.window = args.window if args.window else 6
        self.books_to_read = args.books if args.books else 500

    @staticmethod
    def save_obj(obj, path):
        with open(path, "wb") as wfile:
            pickle.dump(obj, wfile, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, path):
        if os.path.isdir(path):
            for dir_path in os.listdir(path):
                return self.load_obj(os.path.abspath(os.path.join(path, dir_path)))
        else:
            with open(path, "rb") as rfile:
                return pickle.load(rfile)

    @staticmethod
    def line_rule(line):
        length = len(line) if line else -1
        return length > 5 and len([xi for xi in line if xi.isalpha()]) / length >= 0.6

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

    def load_books(self):
        books = []
        for book in pgen.generate_absolute_paths(self.paths_books)[:self.books_to_read]:
            with open(book, "r") as rfile:
                # TODO: split on end of sentence punctuation marks
                #       (dot is not always end of sentence);
                #       think about punctuation in general
                try:
                    lines = [line.strip()
                                 .replace("\n", "")
                                 .replace("'t", " not")
                                 .replace("'s", " is")
                                 .replace("'d", " would")
                                 .replace("'ve", "have")
                             for line in rfile.read().split("\n\n") if self.line_rule(line)]
                    books.extend(lines)
                except:
                    pass

        return books

    def preprocess(self):
        books = self.load_books()
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

        windowed_sentences = self.generate_windowed_sentences(
            tokenized_sentences, word2idx, self.window
        )
        print("windowing done")

        return windowed_sentences, vocabulary, word2idx, idx2word

    def read(self):
        contents = os.listdir('/'.join(self.read_corpus.split("/")[:-1]) + "/")
        # if self.read_corpus and contents and self.read_corpus.split("/")[-1] in contents:
        if self.read_corpus and self.read_corpus.split("/")[-1] in contents:
            corpus_data = self.load_obj(self.read_corpus)
        else:
            corpus_data = CorpusData(*self.preprocess())

            if self.save_corpus:
                self.save_obj(corpus_data, self.save_corpus)

        return corpus_data
