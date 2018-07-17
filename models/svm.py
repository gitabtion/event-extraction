from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
import numpy as np


class SVM(object):
    def __init__(self, train_texts, train_labels, ver_texts, ver_labels, test_texts, test_labels, stopwords):
        self.ver_acc = 0.0
        self.test_acc = 0.0
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.ver_texts = ver_texts
        self.ver_labels = ver_labels
        self.test_texts = test_texts
        self.test_labels = test_labels
        self.stopwords = stopwords
        self.clf = Pipeline([('vect', CountVectorizer(stop_words=stopwords)),
                             ('tfidf', TfidfTransformer()),
                             ('select', SelectKBest(chi2, k=8000)),
                             ('clf', SVC(kernel='linear', C=2))])

    def train(self):
        self.clf.fit(self.train_texts, self.train_labels)

    def verification(self):
        ver_prediction = self.clf.predict(self.ver_texts)
        self.ver_acc = np.mean(ver_prediction == self.ver_labels)

    def test(self):
        test_prediction = self.clf.predict(self.test_texts)
        self.test_acc = np.mean(test_prediction == self.test_labels)
