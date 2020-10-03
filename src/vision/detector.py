import os

import cv2
import numpy as np

from ..helper.functional import Functional
from ..helper.paths_generator import PathsGenerator


class Detector(PathsGenerator, Functional):
    def __init__(self, args):
        import src
        def_pics_paths = os.listdir(os.path.join(self.parent_path(src.SRC_DIR), "data/pics"))

        self.confidence = args.confidence
        self.paths_img = args.paths_img if args.paths_img else def_pics_paths
        self.classes = [x.strip() for x in open(args.labels).read().split()]
        self.net = cv2.dnn.readNetFromDarknet(args.net_config, args.transfer_weights)

    def detect(self):
        return ((label, confidence)
                for img_path in self.flatten(self.generate_absolute_paths(self.paths_img))
                for (label, confidence) in self.predict_label(img_path))

    def predict_label(self, img):
        img = cv2.imread(img)

        layers_names = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # TODO: check if blob's size is important
        # (224, 224),
        self.net.setInput(cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False))

        layerOutputs = self.net.forward(layers_names)

        # TODO: filter top predictions (make sure they regard different object)
        return [(self.classes[int(np.argmax(scores[5:]))], scores[5:][np.argmax(scores[5:])])
                for out in layerOutputs for scores in out if scores[5:][np.argmax(scores[5:])] > self.confidence]
