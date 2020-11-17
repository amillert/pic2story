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
    def generate_windowed_sentences(books, word2idx, window):
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
        books = []
        for book in pgen.generate_absolute_paths(self.paths_books)[:self.books_to_read]:
            with open(book, "r") as rfile:
                # TODO: split on end of sentence punctuation marks
                #       (dot is not always end of sentence);
                #       think about punctuation in general
                try:
                    # TODO: let's take into consideration only a bit longer paragraphs
                    #       and run windowing inside

                    """
                      1. read file
                      2. remove front / end parts
                      3. replace 've, 's, etc.
                      4. find long enough paragraphs, by splitting through chars and find ones with over thresh
                    """

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
                    tmp = [re.split(PUNCT, x) for x in text.split("\n\n")]  # split on punctuation
                    tmp = [[xi for xi in x if (xi.isalpha() and len(xi) > 1) or (not xi.isalpha() and len(xi) == 1)] for x in tmp]
                    tmp = [[xi.strip() for xi in x if xi.strip()] for x in tmp if len(x) > 6]  # remove paragraphs with too few components; leaving us with paragraphs
                    books.append(tmp[20:])  # make sure we begin with text; even tho it's not beginning
                except:
                    pass

        return books  # list of lists of strings (tokens)

    def preprocess(self):
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
        with open("/home/devmood/PycharmProjects/pic2story/cache/windows", "w") as f:
            for w in windowed_sentences:
                f.write(f"{str(w)}\n")

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
