#!/usr/bin/env python

import random

from src.argparser import args
from src.vision.detector import Detector
from src.model.runner import Runner

if __name__ == '__main__':
    detected = Detector(args).detect()
    random.shuffle(detected)
    print(detected)

    runner = Runner(args)
    MODEL = runner.learn()
    print(runner.generate(MODEL, 50, ' '.join(detected)))
