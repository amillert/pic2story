"""
Module implementing the Reader class
"""

from collections import Counter
import os
import pickle
import re

from src.corpus.preprocess.corpus_data import CorpusData
from src.helper import paths_generator as pgen

END_OF_GUTEN = "\n*End of Project Gutenberg(.|\n)*"
FROM_TABLE_OF_CONTENTS = "(^.{0,4}[0-9].*\n*)+(.|\n)*"
AFTER_THE_END = "((THE END)|(The end)|(The End))(.|\n)*"
PUNCT = '(\s|[,.;:?!])'


class Reader:
    """
    Reader class responsible for reading books and building corpus
    """
    def __init__(self, args):
        """
        Constructor of the Reader

        :param args: arguments from the argparser
        """
        self.paths_books = args.paths_books
        self.read_corpus = args.read_corpus
        self.save_corpus = args.save_corpus
        self.window = args.window if args.window else 6
        self.books_to_read = args.books if args.books else 500

    @staticmethod
    def save_obj(obj, path):
        """
        Pickle the provided object

        :param obj: CorpusData
        :param path: str
        :return: None
        """
        with open(path, "wb") as wfile:
            pickle.dump(obj, wfile, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, path):
        """
        Read saved CorpusData object

        :param path: str
        :return: CorpusData
        """
        if os.path.isdir(path):
            for dir_path in os.listdir(path):
                return self.load_obj(os.path.abspath(os.path.join(path, dir_path)))
        else:
            with open(path, "rb") as rfile:
                return pickle.load(rfile)

    @staticmethod
    def generate_windowed_sentences(books, word2idx, window):
        """
        Method responsible for generating subsentences from a sentence
        (adding padding if needed)

        :param books: list[list[list[str]]]
        :param word2idx: dict[(str, int)]
        :param window: int
        :return: list[list[int]]
        """
        res = []
        for book in books:
            for paragraph in book:
                tokens = [word2idx[word] for word in paragraph]
                ile = abs(window - len(tokens))
                if len(tokens) < window:
                    res.append(tokens + [word2idx["<PAD>"]] * ile)
                elif len(tokens) > window:
                    for i in range(0, ile):
                        res.append(tokens[i:i + window])
                else:
                    res.append(tokens)
        return res

    def load_books(self):
        """
        Method reading books from the provided paths, and initially preprocessing them

        Flow:
          1. Book is being read
          2. Regex are being evaluated to try to extract raw content and fix formatting
          3. Split content into paragraphs
          4. Split on punctuation marks
          5. Remove too short paragraphs, or the ones that don't seem like proper text

        Note:
          `except: pass` is intentional; books throwing errors in the face should not be read

        :return: list[list[list[str]]]
        """
        books = []
        for book in pgen.generate_absolute_paths(self.paths_books)[:self.books_to_read]:
            with open(book, "r") as rfile:
                try:
                    text = (
                        re.sub(AFTER_THE_END, "", re.sub(
                            FROM_TABLE_OF_CONTENTS, "", re.sub(
                                END_OF_GUTEN, "", rfile.read())))
                        .replace("-\n", "")  # continuation of the line
                        .replace("-", "")
                        .replace("'t", " not")
                        .replace("'re", " are")
                        .replace("'d", " would")
                        .replace("'s", " is")
                        .replace("'ve", " have"))
                    tmp = [re.split(PUNCT, x) for x in text.split("\n\n")]
                    tmp = [[xi for xi in x if (xi.isalpha() and len(xi) > 1)
                            or (not xi.isalpha() and len(xi) == 1)] for x in tmp]
                    tmp = [[xi.strip() for xi in x if xi.strip()] for x in tmp if len(x) > 6]
                    books.append(tmp[20:])
                except:
                    pass

        return books

    def preprocess(self):
        """
        Method responsible for generating all important corpus objects

        :return: tuple[list[list[int]], list[str], dict[(str, int)], dict[(int, str)]]
        """
        books = self.load_books()
        naive_tokens = [token for book in books for tokens in book for token in tokens]
        print("tokenization done")

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
            books, word2idx, self.window
        )
        print("windowing done")

        return windowed_sentences, vocabulary, word2idx, idx2word

    def read(self):
        """
        Method responsible for running preprocessing, and saving / loading corpus

        :return: CorpusData
        """
        contents = os.listdir('/'.join(self.read_corpus.split("/")[:-1]) + "/")
        # if self.read_corpus and contents and self.read_corpus.split("/")[-1] in contents:
        if self.read_corpus and self.read_corpus.split("/")[-1] in contents:
            corpus_data = self.load_obj(self.read_corpus)
        else:
            corpus_data = CorpusData(*self.preprocess())

            if self.save_corpus:
                self.save_obj(corpus_data, self.save_corpus)

        return corpus_data
