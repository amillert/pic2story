#!/usr/bin/env python

from src.argparser import args
from src.vision.detector import Detector
from src.corpus.reader import Reader
# corpus = Reader(args).read()
from src.model.trainer.runner import Runner

if __name__ == '__main__':
    detected = Detector(args).detect()

    # corpus = Reader(args).read()

    # TODO: ensure compatibility with the learn method in Runner
    # runner = Runner(args, corpus).learn()
    runner = Runner(args).learn()
