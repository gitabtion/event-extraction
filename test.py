from models.svm import SVM
from utils.data_helper import DataHelper


def main():
    data_helper = DataHelper()
    stopwords = data_helper.get_stopwords()
    train_text, train_labels, ver_text, ver_labels, test_text, test_labels = data_helper.get_data_and_labels()
    svm = SVM(train_text, train_labels, ver_text, ver_labels, test_text, test_labels, stopwords)

    svm.train()
    svm.verification()
    print('ver_acc: {:.3}'.format(svm.ver_acc))
    svm.test()
    print('test_acc: {:.3}'.format(svm.test_acc))


if __name__ == '__main__':
    main()
    # test()
