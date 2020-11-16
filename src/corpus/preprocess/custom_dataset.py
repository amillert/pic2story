from src.corpus.reader import Reader

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, args):
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

            # torch.save(self.x_data, os.path.join(root_dir, "x_data.pt"))
            # torch.save(self.y_data, os.path.join(root_dir, "y_data.pt"))
            # print("saving to files done")

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[idx], self.y_data[idx]
