#!/usr/bin/env python

from src.argparser import args
from src.vision.detector import Detector
from src.model.trainer.runner import Runner

if __name__ == '__main__':
    detected = Detector(args).detect()

    runner = Runner(args)
    MODEL = runner.learn()
    print(runner.sample(MODEL, 50, ' '.join(set(detected))))
