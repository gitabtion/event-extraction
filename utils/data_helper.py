from pyltp import Segmentor

import os


class DataHelper(object):
    def __init__(self):
        self.train_labels = []  # 训练集标签
        self.train_texts = []  # 与训练集标签相对应的句子
        self.test_labels = []  # 测试集标签
        self.test_texts = []  # 测试集...
        self.ver_labels = []  # 验证集
        self.ver_texts = []
        self.seg_train_text = []  # 分词后的训练集
        self.seg_test_text = []  # 分词后的测试集
        self.seg_ver_text = []  # 分词后的验证集
        self.stopwords = []
        self.word_set = set()
        self.word_dict = {}
        self._divide()
        self._segment()
        self._gen_word_dict()

    def _divide(self):
        with open('./data/train_set.txt', encoding='utf-8') as f:
            train_set = f.readlines()
        with open('./data/test_set.txt', encoding='utf-8') as f:
            test_set = f.readlines()
        with open('./data/ver_set.txt', encoding='utf-8') as f:
            ver_set = f.readlines()
        for line in train_set:
            t = line.split('\t', 1)  # 分割标签与句子
            self.train_labels.append(t[0])
            self.train_texts.append(t[1].rstrip())
        for line in test_set:
            t = line.split('\t', 1)
            self.test_labels.append(t[0])
            self.test_texts.append(t[1].rstrip())
        for line in ver_set:
            t = line.split('\t', 1)
            self.ver_labels.append(t[0])
            self.ver_texts.append(t[1].rstrip())

    def _segment(self):
        ltp_data_dir = '/Users/abtion/workspace/dataset/ltp_data_v3.4.0'  # ltp模型目录的路径
        cws_model_path = os.path.join(ltp_data_dir, 'cws.model')
        segmentor = Segmentor()  # 初始化实例
        segmentor.load(cws_model_path)  # 加载模型
        for t in self.train_texts:  # 对训练集分词
            temp_words = segmentor.segment(t)
            words = ' '.join(temp_words)
            self.word_set = self.word_set | set(temp_words)
            self.seg_train_text.append(words)
        for t in self.test_texts:  # 对测试集分词
            words = ' '.join(segmentor.segment(t))
            self.seg_test_text.append(words)
        for t in self.ver_texts:  # 对验证集分词
            words = ' '.join(segmentor.segment(t))
            self.seg_ver_text.append(words)
        segmentor.release()

    def get_stopwords(self):
        self.stopwords = [line.rstrip() for line in open('./data/stopwords.txt', encoding='utf-8')]
        return self.stopwords

    def get_data_and_labels(self):
        return (self.seg_train_text, self.train_labels,
                self.seg_ver_text, self.ver_labels,
                self.seg_test_text, self.test_labels)

    def get_word_set(self):
        return self.word_set

    def _gen_word_dict(self):
        _words = list(self.word_set)
        _values = list(i for i in range(len(_words)))
        _wvs = zip(_words, _values)
        self.word_dict = dict((name, value) for name, value in _wvs)
        self.word_dict['<UN>'] = len(_words)

    def get_word_dict(self):
        return self.word_dict

    def line2array(self, seq):
        _words = seq.split(' ')
        _rst = []
        for w in _words:
            _rst.append(self.word_dict[w] if w in self.word_dict.keys() else self.word_dict['<UN>'])
        return _rst
