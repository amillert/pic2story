import torch
import torch.nn as nn


class RNNNet(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNNet, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_size, batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, input_state, previous_state):
        embed = self.embedding(input_state)
        output, state = self.lstm(embed, previous_state)
        logits = self.dense(output)  # what are those ?

        return logits, state

    def zero_state(self, batch_size):
        return tuple(torch.zeros(1, batch_size, self.lstm_size)) * 2
