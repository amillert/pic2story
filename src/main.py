#!/usr/bin/env python

from src.argparser import args
from src.vision.detector import Detector
from src.corpus.reader import Reader
from src.model.trainer.runner import Runner

if __name__ == '__main__':
    detected = Detector(args).detect()

    # TODO: think of a proper logic

    if args.train:
        corpus = Reader(args).read()
    else:
        assert args.load_data is True  # ?

    # TODO: ensure compatibility with the learn method in Runner
    runner = Runner(args).learn()

