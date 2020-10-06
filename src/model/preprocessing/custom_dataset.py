import os
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, args, load_data=False):
        self.args = args
        self.len = 0
        if load_data:
            self.x_data = torch.load(os.path.join("", "x_data.pt")).float()
            self.y_data = torch.load(os.path.join("", "y_data.pt")).float()
            print("loading from files done")
            self.shape = self.x_data.shape
            self.len = self.shape[0]
            self.target_column = ""
        else:
            data = pd.read_csv(os.path.join("", ""), header=0, sep=",", nrows=500000)
            data = self._preprocess_data(data)
            x_data = data.drop([self.target_column], axis=1).values
            y_data = data["is_attributed"].values

            self.len = len(data)

            assert self.len == len(x_data) == len(y_data), \
                f"length mismatch, whole dataset's length is {self.len}, " \
                f"whereas x_data's length is {len(x_data)} and y_data's - {len(y_data)}."

            self.x_data = torch.tensor(x_data, dtype=torch.float32)
            self.y_data = torch.tensor(y_data, dtype=torch.float32)
            self.shape = self.x_data.shape

            # torch.save(self.x_data, os.path.join(root_dir, "x_data.pt"))
            # torch.save(self.y_data, os.path.join(root_dir, "y_data.pt"))
            # print("saving to files done")

    @staticmethod
    def _preprocess_data(data):
        return data

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, 1), torch.zeros(1, 1)
        # return self.x_data[idx], self.y_data[idx]
