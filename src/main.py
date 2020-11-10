#!/usr/bin/env python

from src.argparser import args
from src.vision.detector import Detector
from src.model.trainer.runner import Runner

import random

if __name__ == '__main__':
    detected = Detector(args).detect()
    random.shuffle(detected)

    runner = Runner(args)
    MODEL = runner.learn()
    print(runner.generate(MODEL, 50, ' '.join(detected)))
