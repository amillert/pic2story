"""
Module implementing the WordLSTM class
"""

import os

import numpy as np
import torch
import torch.nn as nn


class WordLSTM(nn.Module):
    """
    WordLSTM class serves as the model for learning process - text generation
    """
    def __init__(self, vocab_size, feature_size, n_hidden=256, n_layers=4, drop_prob=0.4):
        """
        Constructor of the WordLSTM

        :param vocab_size: int
        :param feature_size: int
        :param n_hidden: int
        :param n_layers: int
        :param drop_prob: float
        """
        super().__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.emb_layer = nn.Embedding(vocab_size, feature_size)
        self.lstm = nn.LSTM(feature_size, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.final_layer = nn.Linear(n_hidden, vocab_size)

    def forward(self, x, hidden):
        """
        Method responsible for forward step through the network

        :param x: torch.Tensor
        :param hidden: some state of the LSTM (?)
        :return: tuple[(torch.Tensor, LSTM state)]
        """
        embedded = self.emb_layer(x)

        lstm_output, hidden = self.lstm(embedded, hidden)
        out = self.dropout(lstm_output)
        out = out.reshape(-1, self.n_hidden)

        out = self.final_layer(out)

        return out, hidden

    def init_hidden(self, batch_size):
        """
        Method responsible for initializing weights in hidden layer

        :param batch_size: int
        :return: LSTM state
        """
        weight = next(self.parameters()).data

        # if GPU is available
        if torch.cuda.is_available():
            hidden = (
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda()
            )

        # if GPU is not available
        else:
            hidden = (
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_()
            )

        return hidden

    def save_weights(self):
        """
        Method responsible for saving pretrained weights

        :return: None
        """
        cache_path = os.path.join(os.path.abspath(os.path.curdir), "cache")
        weights_count = len([x for x in os.listdir(cache_path) if "weight" in x])

        with open(os.path.join(cache_path, f"weight{weights_count}"), "wb") as f:
            for embed in self.emb_layer.weight.detach().numpy():
                f.write(' '.join([str(x) for x in embed]).encode("utf-8") + "\n".encode("utf-8"))

    def load_weights(self, path):
        """
        Method responsible for reading pretrained weights

        :return: None
        """
        cache_path = os.path.join(os.path.abspath(os.path.curdir), "cache")
        if path in os.listdir(cache_path):
            with open(os.path.join(cache_path, path), "rb") as f:
                arr = np.array([[float(x) for x in line.strip().decode("utf-8").split()]
                                for line in f.readlines()], dtype=float)

            self.emb_layer.weight.data = torch.nn.Parameter(
                torch.from_numpy(arr).float(), requires_grad=True
            )
