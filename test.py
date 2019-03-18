import random

from torch import nn, optim
from torch.autograd import Variable
import torch

from models.lstm import LSTM
from models.svm import SVM
from utils.data_helper import DataHelper


def main(model):
    data_helper = DataHelper()
    train_text, train_labels, ver_text, ver_labels, test_text, test_labels = data_helper.get_data_and_labels()
    stopwords = data_helper.get_stopwords()
    word_set = data_helper.get_word_set()

    if model == "svm":

        svm = SVM(train_text, train_labels, ver_text, ver_labels, test_text, test_labels, stopwords)

        svm.train()
        svm.verification()
        print('ver_acc: {:.3}'.format(svm.ver_acc))
        svm.test()
        print('test_acc: {:.3}'.format(svm.test_acc))
    elif model == "lstm":

        # 超参数
        batch_size = 1
        num_layers = 1
        num_directions = 1
        embedding_size = 300
        hidden_size = 1
        words_length = len(word_set)
        learning_rate = 0.0001
        num_epochs = 2

        lstm = LSTM(words_length, embedding_size, hidden_size, num_layers, num_directions, batch_size)
        optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            c = list(zip(train_text, train_labels))
            random.shuffle(c)
            data_train = [i[0] for i in c]
            label_train = [i[1] for i in c]
            count = 0

            for d, l in zip(data_train, label_train):
                count += 1
                if count % 100 == 0:
                    print(epoch, count, loss)

                seq = data_helper.line2array(d)

                sent = Variable(torch.tensor(seq))
                target = Variable(torch.tensor([int(l)]))

                optimizer.zero_grad()
                output = lstm(sent)

                loss_func = nn.CrossEntropyLoss()
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()


def test():
    return


if __name__ == '__main__':
    main("lstm")
    # test()
