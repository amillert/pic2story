import os
from typing import List, Tuple, Iterable

from ..helper import functional as fct
from ..helper import paths_generator as pgen


class Reader:
    def __init__(self, args):
        import src
        def_books_paths = os.listdir(os.path.join(pgen.parent_path(src.SRC_DIR), "data/kidsbooks"))

        self.paths_books = args.paths_books if args.paths_books else def_books_paths
        self.ctx = args.ngrams

    @staticmethod
    def line_rule(line: str) -> bool:
        ln = len(line) if len(line) else -1
        return ln > 5 and len([xi for xi in line if xi.isalpha()]) / ln >= 0.6

    def generate_pad(self, n: int) -> List[str]:
        return ["<PAD>"] * (self.ctx - n)

    def generate_pads(self, left: List[str], right: List[str]) -> Tuple[List[str], List[str]]:
        return self.generate_pad(len(left)) + left, right + self.generate_pad(len(right))

    def generate_left_idx(self, i: int) -> List[int]:
        return [x for x in range(i - self.ctx, i) if x >= 0]

    def generate_right_idx(self, i: int, ln: int) -> List[int]:
        return [x for x in range(i + 1, i + 1 + self.ctx) if x < ln]

    def generate_idx(self, i: int, s: List[str]) -> Tuple[List[str], List[str]]:
        return [s[i] for i in self.generate_left_idx(i)], [s[i] for i in self.generate_right_idx(i, len(s))]

    def extract_context(self, i: int, sentence: List[str]) -> Tuple[List[str], List[str]]:
        return self.generate_pads(*self.generate_idx(i, sentence))

    # TODO: (just as an example); check what will be needed for the generation
    def generate_ngrams(self, sentence: List[str]) -> Iterable[Tuple[str, List[str]]]:
        for i, target in enumerate(sentence):
            left_ctx, right_ctx = self.extract_context(i, sentence)
            yield target, left_ctx + right_ctx

    def read(self):
        ngrams = []

        # TODO: check whether distinguishing one book from the other matters
        #       in terms of text generation (same with chapters, etc.)

        for book in fct.flatten(pgen.generate_absolute_paths(self.paths_books)):
            with open(book) as w:

                # TODO: split on end of sentence punctuation marks (dot is not always end of sentence);
                #       think about punctuation in general

                lines = [line.strip().replace("\n", "") for line in w.read().split("\n\n") if self.line_rule(line)]
                naively_tokenized = [sentence.split() for sentence in ' '.join(lines).split(".")]

                ngrams.append([self.generate_ngrams(sentence) for sentence in naively_tokenized if len(sentence)])

        assert len(ngrams) == len(self.paths_books)

        return ngrams
