"""
Module responsible for implementing Runner class
"""

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.corpus.preprocess.custom_dataset import CustomDataset
from src.model.word_LSTM import WordLSTM


class Runner:
    """
    Runner class responsible for running the learning process, prediction, etc.
    """
    def __init__(self, args):
        """
        Constructor of the Runner class

        :param args: arguments from the argparser
        """
        self.args = args
        # TODO: use these (2)
        self.load_x = False
        self.load_y = False

        # hyper-parameters
        self.batch_size = args.batch_size
        self.eta = args.eta
        self.epochs = args.epochs
        self.hidden = args.hidden
        self.layers = args.layers
        self.drop_prob = args.drop_prob

        # read / write embedding layer's weights
        self.load_pretrained = self.args.load_pretrained
        self.save_weights = self.args.save_weights

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataset = CustomDataset(args)
        self.corpus = self.dataset.corpus

    def generate(self, size):
        """
        Method responsible for the text-generation

        :param model: WordLSTM
        :param size: int
        :param prime: list[str]
        :return: WordLSTM
        """
        # push to GPU
        # model.to(self.device)

        _, feature_size = self.dataset.shape

        model = WordLSTM(
            len(self.corpus.vocabulary) + 1,
            feature_size,
            self.hidden,
            self.layers,
            0.3
        )

        if self.load_pretrained:
            import os
            cache_path = os.path.join(os.path.abspath(os.path.curdir), "cache")
            path = "/home/devmood/PycharmProjects/pic2story/cache/model"
            if "model" in os.listdir(cache_path):
                model.load_state_dict(torch.load(path))
                # model.load_state_dict(path)
                print("WCZYTANO!!!!!!!!\n"*4)

        # path = "/home/devmood/PycharmProjects/pic2story/cache/model"
        # model.load_state_dict(torch.load(path))

        # change require grad only for relevant weights
        for name, param in model.named_parameters():
            if "final" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in model.named_parameters():
            print(name, ':', param.requires_grad)

        tokens = [token for token in ["juicy", "sister", "cars", "not"]]
        print(tokens)

        def predict():
            model.eval()
            state_h, state_c = model.init_hidden(len(tokens))

            for i in range(0, size):
                X = torch.tensor([[self.corpus.word2idx[w]] for w in tokens[i:]])
                y_pred, (state_h, state_c) = model(X, (state_h, state_c))

                last_word_logits = y_pred[0]  # [-1]
                p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
                word_index = np.random.choice(len(last_word_logits), p=p)
                tokens.append(self.corpus.idx2word[word_index])

            return tokens

        tmp = predict()
        recent = ""
        res = ""
        for x in tmp:
            if x == recent:
                continue
            if x.isalpha():
                res += " "
            res += x
            if x in ".!?":
                res = res[:-1] + "\n"
            recent = x

        return res

    def learn(self):
        """
        Method performing the whole learning process

        :return: WordLSTM
        """
        data_size, feature_size = self.dataset.shape

        print(f"Amount of data read: {data_size}")
        batches = DataLoader(
            dataset=self.dataset,
            drop_last=True,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6
        )
        print('Creating batches done')

        # TODO: load from the argparser (2 ?)
        model = WordLSTM(
            len(self.corpus.vocabulary) + 1,
            feature_size,
            self.hidden,
            self.layers,
            0.3
        )

        if self.load_pretrained:
            import os
            cache_path = os.path.join(os.path.abspath(os.path.curdir), "cache")
            path = "/home/devmood/PycharmProjects/pic2story/cache/model"
            if "model" in os.listdir(cache_path):
                model.load_state_dict(torch.load(path))

        # change require grad only for relevant weights
        for name, param in model.named_parameters():
            if "final" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.eta)  # 0.5

        # TODO: what it's for?
        # clip = 1

        num_batches = int(data_size / self.batch_size)
        print(f"upcoming num batches: {num_batches}")

        model.train()

        for epoch in range(1, self.epochs + 1):
            # tick = time.time()
            loss_total = 0.0
            batch_count = 0

            state_h = model.init_hidden(self.batch_size)

            for X, y in batches:
                batch_count += 1

                # model.train()

                # move to the GPU if possible
                # inputs, targets = X.to(self.device), y.to(self.device)
                # inputs, targets = X.to(self.device), y.to(self.device)

                hidden = tuple([each.data for each in state_h])

                optimizer.zero_grad()

                output, _ = model(X, hidden)

                loss = criterion(output, y.view(-1))

                loss_total += loss.item()

                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                # nn.utils.clip_grad_norm_(model.parameters(), clip)

                optimizer.step()

                if not batch_count % 10:
                    print(f"Batch {batch_count}/{num_batches}: loss ~> {loss.item()}, mean ~> {loss_total / batch_count}")
                    # print('mean loss {0:.4f}'.format(loss_total / batch_count))

            print("~~~~~~~~~~~~~~~~~~")
            print(f"Epoch: {epoch} out of {self.epochs}")
            # print(f"Time per epoch: {time.time() - tick}")
            print(f"Mean loss: {loss_total / num_batches:.4f}")
            print(f"Total loss: {loss_total:.4f}")
            print("~~~~~~~~~~~~~~~~~~")

            if self.save_weights:
                path = "/home/devmood/PycharmProjects/pic2story/cache/model"
                torch.save(model.state_dict(), path)
                # model.save_weights()

        return model
