#!/usr/bin/env python

from src.argparser import args
from src.vision.detector import Detector
from src.corpus.reader import Reader


if __name__ == '__main__':
    # if args.train:
    #     pass
    detected = Detector(args).detect()
    corpus = Reader(args).read()
    print("main", corpus)
