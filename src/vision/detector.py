import os

import cv2
import numpy as np

from ..helper import functional as fct
from ..helper import paths_generator as pgen


class Detector:
    def __init__(self, args):
        import src
        def_pics_paths = os.listdir(os.path.join(pgen.parent_path(src.SRC_DIR), "data/pics"))

        self.confidence = args.confidence
        self.paths_img = args.paths_img if args.paths_img else def_pics_paths
        self.classes = [x.strip() for x in open(args.labels).readlines()]
        self.net = cv2.dnn.readNetFromDarknet(args.net_config, args.transfer_weights)

    def detect(self):
        # allows duplicates, in case detected in different pictures
        # TODO: think whether that's desired
        return [label for img_path in fct.flatten(pgen.generate_absolute_paths(self.paths_img))
                for label in set(self.predict_label(img_path))]

    def predict_label(self, img):
        img = cv2.imread(img)
        layers_names = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.net.setInput(cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), (0, 0, 0), True, crop=False))

        layerOutputs = self.net.forward(layers_names)

        return [self.classes[int(np.argmax(scores[5:]))] for out in layerOutputs for scores in out
                if scores[5:][np.argmax(scores[5:])] > self.confidence]
