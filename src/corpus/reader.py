from collections import Counter
import os
import pickle
import re

from src.corpus.preprocess.corpus_data import CorpusData
from src.helper import paths_generator as pgen

FROM_TABLE_OF_CONTENTS = "(^.{0,4}[0-9].*\n*)+(.|\n)*"
AFTER_THE_END = "((THE END)|(The end)|(The End))(.|\n)*"


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
    def generate_windowed_sentences(tokenized, word2idx, window):
        res = []
        tokens = [word2idx[s] for s in tokenized]
        for i in range(len(tokens) - window):
            res.append(tokens[i:i+window])

        print(res[-2:])
        return res

    def load_books(self):
        books = []
         #for book in pgen.generate_absolute_paths(self.paths_books):
        for book in pgen.generate_absolute_paths(self.paths_books)[:self.books_to_read]:
            with open(book, "r") as rfile:
                # TODO: split on end of sentence punctuation marks
                #       (dot is not always end of sentence);
                #       think about punctuation in general
                try:
                    text = rfile.read()
                    # if "THE END" not in text or "Contents" not in text:
                    #     continue
                    # else:
                    books.append(re.sub(
                        "\[([A-Za-z0-9.,;]| )+\]", "", re.sub(
                            AFTER_THE_END, "", re.sub(
                                FROM_TABLE_OF_CONTENTS, "", ' '.join(text.split()), 1))
                            .replace("-\n", "")
                            .replace("'t", " not")
                            .replace("'d", " would")
                            .replace("'s", " is")
                            .replace("'ve", " have")))
                except:
                    pass

        return [xi.strip() for x in re.split('(\s"|[,.;:?!"-]\s)', ' '.join(books)) for xi in x.split()]

    def preprocess(self):
        tokenized_books = self.load_books()
        print("tokenization done")

        word2freq = Counter(tokenized_books)
        print("word2freq done")
        vocabulary = ["<PAD>"] + list(word2freq.keys())
        print("vocab done")
        word2idx = {"<PAD>": 0}
        word2idx.update({token: i + 1 for i, token in enumerate(vocabulary)})
        print("word2idx done")
        idx2word = {v: k for k, v in word2idx.items()}
        print("index done")

        windowed_sentences = self.generate_windowed_sentences(
            tokenized_books, word2idx, self.window
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
