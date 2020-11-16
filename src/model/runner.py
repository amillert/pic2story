from src.corpus.preprocess.custom_dataset import CustomDataset
from src.model.word_LSTM import WordLSTM

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Runner:
    def __init__(self, args):
        self.args = args
        # TODO: use these
        self.load_x = False
        self.load_y = False

        # hyper-parameters
        self.batch_size = args.batch_size
        self.eta = args.eta
        self.epochs = args.epochs
        self.hidden = args.hidden
        self.layers = args.layers
        self.drop_prob = args.drop_prob

        self.load_pretrained = self.args.load_pretrained
        self.save_weights = self.args.save_weights

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataset = CustomDataset(args)
        self.corpus = self.dataset.corpus

    def predict(self, model, token, h=None):
        # TODO (1.2): remove this part after 1.1 is satisfied
        # print(self.corpus.vocabulary)
        x = np.array([[token]])

        inputs = torch.from_numpy(x)
        inputs.to(self.device)

        h = tuple([each.data for each in h])
        out, h = model(inputs, h)
        p = F.softmax(out, dim=1).data

        p = p.cpu()
        p = p.numpy()
        p = p.reshape(p.shape[1], )

        # get indices of top 3 values
        top_n_idx = p.argsort()[-3:][::-1]
        # randomly select one of the three indices
        sampled_token_index = top_n_idx[random.sample([0, 1, 2], 1)[0]]

        return sampled_token_index, h

    def generate(self, model, size, prime):
        # push to GPU
        model.to(self.device)
        model.load_weights(self.load_pretrained)
        model.eval()

        # batch size is 1
        h = model.init_hidden(1)

        # TODO (1.1): dataset is too small, so the detected labels
        #             are not even in the vocabulary yielding error
        # tokens = [self.corpus.word2idx[token]
        #           for token in prime.split() if not token is "bicycle"]
        tokens = [self.corpus.word2idx[token]
                  for token in ["wife", "train", "vinegar", "houses"]]

        # predict next token
        for t in tokens:
            token, h = self.predict(model, t, h)
        tokens.append(token)

        # predict subsequent tokens
        for i in range(size - 1):
            token, h = self.predict(model, tokens[-1], h)
            tokens.append(token)

        return ' '.join([self.corpus.idx2word[token] for token in tokens])

    def learn(self):
        data_size, feature_size = self.dataset.shape

        print(f"Amount of data read: {data_size}")
        batches = DataLoader(
            dataset=self.dataset, drop_last=True, batch_size=self.batch_size, shuffle=True, num_workers=6
        )
        print('Creating batches done')

        # TODO: load from the argparser
        model = WordLSTM(len(self.corpus.vocabulary) + 1, feature_size, self.hidden, self.layers,  0.3)

        # if self.load_pretrained:
        model.load_weights(self.load_pretrained)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.eta)

        # TODO: what it's for?
        CLIP = 1

        # print(data_size, self.batch_size)
        num_batches = int(data_size / self.batch_size)
        print(f"upcoming num batches: {num_batches}")

        # for epoch in range(self.epochs):
        for epoch in range(1):
            tick = time.time()
            loss_total = 0.0
            batch_count = 0

            state_h = model.init_hidden(self.batch_size)

            for X, y in batches:
                batch_count += 1

                model.train()

                # move to the GPU if possible
                # TODO: ensure X and y are Tensors
                inputs, targets = X.to(self.device), y.to(self.device)

                h = tuple([each.data for each in state_h])

                model.zero_grad()

                output, h = model(inputs, h)

                loss = criterion(output, targets.view(-1))

                loss_total += loss.item()

                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(model.parameters(), CLIP)

                optimizer.step()

                if num_batches % batch_count == 100:
                    print(f"Batch {batch_count} of epoch {epoch + 1}")
                    print('mean loss {0:.4f}'.format(loss_total / batch_count))

            print("~~~~~~~~~~~~~~~~~~")
            print(f"Epoch: {epoch + 1} out of {self.epochs}")
            print(f"Time per epoch: {time.time() - tick}")
            print(f"Mean loss: {loss_total / num_batches:.4f}")
            print(f"Total loss: {loss_total:.4f}")
            print("~~~~~~~~~~~~~~~~~~")

        # TODO: speedy implementation, to be potentially fixed
        if self.save_weights:
            model.save_weights()

        return model
