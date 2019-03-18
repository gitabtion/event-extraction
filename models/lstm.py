import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, words_length, embedding_size, hidden_size, num_layers, num_directions, batch_size):
        super(LSTM, self).__init__()

        self.words_length = words_length
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.batch_size = batch_size
        self.embedding = nn.Embedding(words_length, embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.batch_size, batch_first=True)
        self.ln1 = nn.Linear(hidden_size, 128)
        self.ln2 = nn.Linear(128, 9)

    def forward(self, x):
        w2v = self.embedding(x)
        w2v = w2v.view(1, -1, self.embedding_size)

        h0 = Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size))
        out, _ = self.lstm(w2v, (h0, c0))

        out = F.sigmoid(self.ln1(out[:, -1, :]))
        out = F.sigmoid(self.ln2(out))

        return out
