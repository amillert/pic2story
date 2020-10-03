from unittest import TestCase

from src.helper import functional as fct
from src.helper import paths_generator as pgen
from src.argparser import args
from src.vision.detector import Detector


class TestHelper(TestCase):
    def setUp(self):
        self.detector = Detector(args)

    def test_generate_paths(self):
        self.assertEqual(len(pgen.generate_absolute_paths(["./data/pics"])), 2)
        self.assertEqual(len(pgen.generate_absolute_paths([""])), 0)
        self.assertEqual(len(pgen.generate_absolute_paths([])), 0)
        self.assertEqual(len(pgen.generate_absolute_paths(["./data/pics/cat.jpg"])), 1)

    def test_flatten_generate_paths(self):
        self.assertEqual(len([*fct.flatten(pgen.generate_absolute_paths(["../data/pics", "../data/pics/cat.jpg"]))]), 3)
        self.assertEqual(len([*fct.flatten(pgen.generate_absolute_paths([]))]), 0)
        self.assertEqual(len([*fct.flatten(pgen.generate_absolute_paths([""]))]), 0)
        self.assertEqual(len([*fct.flatten(pgen.generate_absolute_paths(["../data/pics"]))]), 2)
        self.assertEqual(len([*fct.flatten(pgen.generate_absolute_paths(["../data/pics/cat.jpg"]))]), 1)

    def test_unique_generate_paths(self):
        self.assertEqual(len(fct.unique(fct.flatten(
            pgen.generate_absolute_paths(["../data/pics", "../data/pics/cat.jpg"])))), 2)
        self.assertEqual(len(fct.unique(fct.flatten(
            pgen.generate_absolute_paths(["../data/pics"])))), 2)
        self.assertEqual(len(fct.unique(fct.flatten(
            pgen.generate_absolute_paths([""])))), 0)
        self.assertEqual(len(fct.unique(fct.flatten(
            pgen.generate_absolute_paths([])))), 0)
        self.assertEqual(len(fct.unique(fct.flatten(
            pgen.generate_absolute_paths(["../data/pics/cat.jpg"])))), 1)
