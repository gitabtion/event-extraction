import random

from torch import nn, optim
from torch.autograd import Variable
import torch
import torch.utils.data as Data
import numpy as np

from models.lstm import LSTM
from models.svm import SVM
from utils.data_helper import DataHelper


def main(model):
    if model == "svm":
        train_svm()

    elif model == "lstm":
        train_lstm()


def train_svm():
    data_helper = DataHelper()
    train_text, train_labels, ver_text, ver_labels, test_text, test_labels = data_helper.get_data_and_labels()
    stopwords = data_helper.get_stopwords()

    svm = SVM(train_text, train_labels, ver_text, ver_labels, test_text, test_labels, stopwords)

    svm.train()
    svm.verification()
    print('ver_acc: {:.3}'.format(svm.ver_acc))
    svm.test()
    print('test_acc: {:.3}'.format(svm.test_acc))


def train_lstm():
    batch_size = 100
    num_layers = 3
    num_directions = 2
    embedding_size = 100
    hidden_size = 64
    learning_rate = 0.0001
    num_epochs = 5

    data_helper = DataHelper()
    train_text, train_labels, ver_text, ver_labels, test_text, test_labels = data_helper.get_data_and_labels()
    word_set = data_helper.get_word_set()
    vocab = data_helper.get_word_dict()
    words_length = len(word_set) + 2

    lstm = LSTM(words_length, embedding_size, hidden_size, num_layers, num_directions, batch_size)
    X = [[vocab[word] for word in sentence.split(' ')] for sentence in train_text]
    X_lengths = [len(sentence) for sentence in X]
    pad_token = vocab['<PAD>']
    longest_sent = max(X_lengths)
    b_size = len(X)
    padded_X = np.ones((b_size, longest_sent)) * pad_token
    for i, x_len in enumerate(X_lengths):
        sequence = X[i]
        padded_X[i, 0:x_len] = sequence[:x_len]

    x = Variable(torch.tensor(padded_X)).long()
    y = Variable(torch.tensor(list(int(i) for i in train_labels)))
    dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            output = lstm(batch_x)
            temp = torch.argmax(output, dim=1)
            correct = 0
            for i in range(batch_size):
                if batch_y[i] == temp[i]:
                    correct += 1

            loss = loss_func(output, batch_y)
            print('epoch: {0}, step: {1}, loss: {2}, train acc: {3}'.format(epoch, step, loss, correct / batch_size))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ver_lstm(lstm, ver_text, ver_labels, vocab, batch_size)
    test_lstm(lstm, test_text, test_labels, vocab, batch_size)


def ver_lstm(lstm, ver_texts, ver_labels, vocab, batch_size):
    X = [[vocab[word] if word in vocab else vocab['<UN>'] for word in sentence.split(' ')] for sentence in ver_texts]
    X_lengths = [len(sentence) for sentence in X]
    pad_token = vocab['<PAD>']
    longest_sent = max(X_lengths)
    b_size = len(X)
    padded_X = np.ones((b_size, longest_sent)) * pad_token
    for i, x_len in enumerate(X_lengths):
        sequence = X[i]
        padded_X[i, 0:x_len] = sequence[:x_len]

    x = Variable(torch.tensor(padded_X)).long()
    y = Variable(torch.tensor(list(int(i) for i in ver_labels)))
    dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    correct = 0
    loss = 0.0
    for step, (batch_x, batch_y) in enumerate(loader):
        output = lstm(batch_x)
        temp = torch.argmax(output, dim=1)
        for i in range(batch_size):
            if batch_y[i] == temp[i]:
                correct += 1

        loss_func = nn.CrossEntropyLoss()
        loss += loss_func(output, batch_y)
    print('loss: {0}, ver acc: {1}'.format(loss * batch_size / len(ver_labels), correct / len(ver_labels)))


def test_lstm(lstm, test_texts, test_labels, vocab, batch_size):
    X = [[vocab[word] if word in vocab else vocab['<UN>'] for word in sentence.split(' ')] for sentence in test_texts]
    X_lengths = [len(sentence) for sentence in X]
    pad_token = vocab['<PAD>']
    longest_sent = max(X_lengths)
    b_size = len(X)
    padded_X = np.ones((b_size, longest_sent)) * pad_token
    for i, x_len in enumerate(X_lengths):
        sequence = X[i]
        padded_X[i, 0:x_len] = sequence[:x_len]

    x = Variable(torch.tensor(padded_X)).long()
    y = Variable(torch.tensor(list(int(i) for i in test_labels)))
    dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    correct = 0
    loss = 0.0
    for step, (batch_x, batch_y) in enumerate(loader):
        output = lstm(batch_x)
        temp = torch.argmax(output, dim=1)
        for i in range(batch_size):
            if batch_y[i] == temp[i]:
                correct += 1

        loss_func = nn.CrossEntropyLoss()
        loss += loss_func(output, batch_y)
    print('loss: {0}, test acc: {1}'.format(loss * batch_size / len(test_labels), correct / len(test_labels)))


def test():
    data_helper = DataHelper()
    train_text, train_labels, ver_text, ver_labels, test_text, test_labels = data_helper.get_data_and_labels()
    labels = list(int(i) for i in train_labels)
    wts = np.bincount(labels)
    print(wts)


if __name__ == '__main__':
    main("lstm")
