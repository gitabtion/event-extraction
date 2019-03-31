import torch
from torch import nn


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
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)
        self.ln1 = nn.Linear(hidden_size, 128)
        self.ln2 = nn.Linear(128, 64)
        self.ln3 = nn.Linear(64, 9)

    def forward(self, x):
        w2v = self.embedding(x)
        w2v = w2v.view(self.batch_size, -1, self.embedding_size)

        out, _ = self.lstm(w2v, None)

        out = torch.relu(self.ln1(out[:, -1, :]))
        out = torch.relu(self.ln2(out))
        out = self.ln3(out)

        return out
