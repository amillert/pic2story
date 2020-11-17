"""
Module implementing the CustomDataset(Dataset) class
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.corpus.reader import Reader


class CustomDataset(Dataset):
    """
    CustomDataset class needed to create mini-batches using PyTorch
    """
    def __init__(self, args):
        """
        Constructor of the CustomDataset

        :param args: arguments from the argparser
        """
        self.args = args
        self.len = 0
        if args.load_data:
            self.x_data = torch.load(os.path.join("", "x_data.pt")).float()
            self.y_data = torch.load(os.path.join("", "y_data.pt")).float()
            print("loading from files done")
            self.shape = self.x_data.shape
            self.len = self.shape[0]
        else:
            corpus = Reader(args).read()
            self.corpus = corpus
            data = corpus.windowed

            x_data = np.array(data[1:], dtype=float)
            y_data = np.array(data[:-1], dtype=float)

            self.x_data = torch.tensor(x_data, dtype=torch.long)
            self.y_data = torch.tensor(y_data, dtype=torch.long)
            self.shape = self.y_data.shape

            self.len = len(x_data)

    def __len__(self):
        """
        Method responsible for providing length of the dataset

        :return: int
        """
        return self.len

    def __getitem__(self, idx):
        """
        Method responsible for indexing a given tensor pairs

        :param idx: int
        :return: tuple[(torch.Tensor, torch.Tensor)]
        """
        return self.x_data[idx], self.y_data[idx]
