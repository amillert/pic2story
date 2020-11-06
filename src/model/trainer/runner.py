from ..preprocessing.custom_dataset import CustomDataset
from ..rnn_net import RNNNet
from ..word_LSTM import WordLSTM

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class Runner:
    def __init__(self, args):
        # TODO: timer and tqdm as a argument in argparser
        self.args = args
        # self.DATASET = CustomDataset(self.args, corpus)
        self.track = True
        self.load_x = False
        self.load_y = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, MODEL, token, h=None):
        x = np.array([[self.corpus.idx2word[token]]])
        inputs = torch.from_numpy(x)

        inputs.to(self.device)

        h = tuple([each.data for each in h])

        out, h = MODEL(inputs, h)
        p = F.softmax(out, dim=1).data

        p = p.cpu()

        p = p.numpy()
        p = p.reshape(p.shape[1], )

        # get indices of top 3 values
        top_n_idx = p.argsort()[-3:][::-1]

        # randomly select one of the three indices
        sampled_token_index = top_n_idx[random.sample([0, 1, 2], 1)[0]]

        # return the encoded value of the predicted char and the hidden state
        return self.corpus.idx2word[sampled_token_index], h

    # function to generate text
    def sample(self, MODEL, size, prime='it is'):

        # push to GPU
        MODEL.to(self.device)

        MODEL.eval()

        # batch size is 1
        h = MODEL.init_hidden(1)

        tokens = prime.split()

        # predict next token
        for t in prime.split():
            token, h = self.predict(MODEL, t, h)

        tokens.append(token)

        # predict subsequent tokens
        for i in range(size - 1):
            token, h = self.predict(MODEL, tokens[-1], h)
            tokens.append(token)

        return ' '.join(tokens)

    def learn(self):
        DATASET = CustomDataset(self.args)
        self.corpus = DATASET.corpus

        DATA_SIZE, FEATURE_SIZE = DATASET.shape
        BATCH_SIZE = 500

        # print(DATA_SIZE, FEATURE_SIZE)
        # exit(12)
        # assert not DATA_SIZE % BATCH_SIZE
        #     f"BATCH_SIZE must be a divisor of the DATASIZE, else the model will not have a proper layer size set. "
        #         f"Given BATCH_SIZE: {BATCH_SIZE} and DATA_SIZE: {DATA_SIZE}. For Your reference, use one of these: "
        #         f"{[x for x in range(DATA_SIZE) if 1 < x <= 1024 and not DATA_SIZE % x]}")

        # DIMENSIONS (H_DIMS, F_DIMS, etc.)

        print(f"Amount of data read: {DATA_SIZE}")
        BATCHES = DataLoader(dataset=DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
        print('Creating batches done')

        # TODO: load from the argparser
        # MODEL = RNNNet(BATCH_SIZE, 0, 0, 10)
        MODEL = WordLSTM(len(DATASET.corpus.vocabulary), 256, 20, 0.3)
        EPOCHS = 10
        ETA = 5
        # MOMENTUM = 0

        CRITERION = nn.CrossEntropyLoss()
        OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=ETA)  # , momentum=MOMENTUM)
        # OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=ETA)

        # TODO: what it's for?
        CLIP = 1

        NUM_BATCHES = np.ceil(DATA_SIZE / BATCH_SIZE)
        print(f"Upcoming batches: {NUM_BATCHES}")

        for epoch in tqdm(range(EPOCHS)):
            correctly_classified, total_classified = 0.0, 0.0
            tick = time.time()
            loss_total = 0.0
            batch_count = 0

            state_h = MODEL.init_hidden(BATCH_SIZE)

            for X, y in BATCHES:
                batch_count += 1

                MODEL.train()

                # move to the GPU if possible
                # TODO: ensure X and y are Tensors
                inputs, targets = X.to(self.device), y.to(self.device)

                h = tuple([each.data for each in state_h])

                # OPTIMIZER.zero_grad()
                MODEL.zero_grad()

                output, h = MODEL(inputs, h)

                loss = CRITERION(output, targets.view(-1))

                # state_h = state_h.detach()
                # state_c = state_h.detach()

                print(f"Batch {batch_count} loss: {loss}, mean loss: {loss_total / batch_count}")
                loss_total += loss.item()

                # OPTIMIZER.zero_grad()
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(MODEL.parameters(), CLIP)

                OPTIMIZER.step()

                if not batch_count % 10 and batch_count > 0:
                    print(f"Batch {batch_count} of epoch {epoch + 1}")
                    print('mean loss {0:.4f}'.format(loss_total / batch_count))

            tock = time.time()
            print("~~~~~~~~~~~~~~~~~~")
            print(f"Epoch: {epoch + 1} out of {EPOCHS}")
            print(f"Time per epoch: {tock - tick}")
            print(f"Mean loss: {loss_total / NUM_BATCHES}")
            print(f"Total loss: {loss_total}")
            print("~~~~~~~~~~~~~~~~~~")
            print("@@@@@@@@@@@@@@@@@@")
            print(f"Accuracy of the model: {correctly_classified / total_classified * 100.0}%")
            print("@@@@@@@@@@@@@@@@@@")

            # TODO: saving pretrained weights

        return MODEL
