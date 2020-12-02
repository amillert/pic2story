#!/usr/bin/env python
"""
Main of the whole project
"""

import random

from src.argparser import args
from src.vision.detector import Detector
from src.model.runner import Runner
from src.model.evaluate import score

if __name__ == '__main__':
    detected, synonyms = Detector(args).detect()
    random.shuffle(detected)
    print(detected)

    runner = Runner(args)
    MODEL = runner.learn()
    generated = runner.generate(50, ' '.join(detected))

    score(generated, synonyms)
