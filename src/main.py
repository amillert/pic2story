#!/usr/bin/env python

from src.argparser import args
from src.vision.detector import Detector


if __name__ == '__main__':
    # if args.train:
    #     pass
    detected = Detector(args).detect()
