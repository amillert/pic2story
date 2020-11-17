"""
Module implementing the Detector class
"""
import os

import cv2
import numpy as np

import src
from src.helper import paths_generator as pgen


class Detector:
    """
    Detector used for detecting objects in the pictures using YOLO
    """
    def __init__(self, args):
        """
        Constructor of the Detector

        :param args: arguments from the argparser
        """
        def_pics_paths = os.listdir(os.path.join(pgen.parent_path(src.SRC_DIR), "data/pics"))
        self.confidence = args.confidence
        self.paths_img = args.paths_img if args.paths_img else def_pics_paths
        self.classes = [x.strip() for x in open(args.labels).readlines()]
        self.net = cv2.dnn.readNetFromDarknet(args.net_config, args.transfer_weights)

    @staticmethod
    def generate_absolute_paths(paths):
        """
        Method for generating absolute paths

        :param paths: list[str] ?
        :return: list[list[str]]
        """
        abs_paths = [os.path.abspath(path) if not os.path.isabs(path) else path for path in paths]
        return [[os.path.join(abs_path, pic) for pic in os.listdir(abs_path)]
                if os.path.isdir(abs_path) else [abs_path] for abs_path in abs_paths]

    def detect(self):
        """
        Method responsible for detecting labels in pictures

        :return: list[str]
        """
        return list({label for img_path in
                     [xi for x in self.generate_absolute_paths(self.paths_img) for xi in x]
                     for label in set(self.predict_label(img_path))})

    def predict_label(self, img):
        """
        Method responsible for predicting the objects

        :param img: str
        :return: list[list[int]] ?
        """
        img = cv2.imread(img)
        layers_names = [self.net.getLayerNames()[i[0] - 1]
                        for i in self.net.getUnconnectedOutLayers()]

        self.net.setInput(
            cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), (0, 0, 0), True, crop=False)
        )

        layer_outputs = self.net.forward(layers_names)

        return [self.classes[int(np.argmax(scores[5:]))] for out in layer_outputs
                for scores in out if scores[5:][np.argmax(scores[5:])] > self.confidence]
