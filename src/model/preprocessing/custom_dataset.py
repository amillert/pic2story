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
            self.target_column = ""
            # doesn't matter for now
        else:
            # TODO: make this part work first
            corpus = Reader(args).read()
            data = corpus.ngrams

            targets = np.array([target for target, _ in data])
            contexts = np.array([context for _, context in data])

            # data = pd.read_csv(os.path.join("", ""), header=0, sep=",", nrows=500000)
            # data = self._preprocess_data(data)
            # x_data = data.drop([self.target_column], axis=1).values
            # y_data = data["is_attributed"].values
            # data = corpus.ngrams

            self.x_data = torch.tensor(targets, dtype=torch.float32)
            self.y_data = torch.tensor(contexts, dtype=torch.float32)
            self.shape = self.y_data.shape

            self.len = len(data)

            assert self.len == len(self.x_data) == len(self.y_data), \
                f"length mismatch, whole dataset's length is {self.len}, " \
                f"whereas x_data's length is {len(self.x_data)} and y_data's - {len(self.y_data)}."

            # torch.save(self.x_data, os.path.join(root_dir, "x_data.pt"))
            # torch.save(self.y_data, os.path.join(root_dir, "y_data.pt"))
            # print("saving to files done")

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # return torch.zeros(1, 1), torch.zeros(1, 1)
        return self.x_data[idx], self.y_data[idx]
