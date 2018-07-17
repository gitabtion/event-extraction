# text classification
text classification by some machine learning algorithm.

|model|accuracy|
|:-:|:-:|
|SVC with Linear kernel|0.718|

## download

> $git clone git@github.com:gitabtion/text-classification.git

## getting start

> $cd text-classification

> $python3 test.py




## dictionary
```
├── LICENSE
├── README.md
├── data
│   ├── stopwords.txt           # stopword
│   ├── test_set.txt            # testing set
│   ├── test_set_name.txt      
│   ├── train_set.txt           # trainning set
│   └── ver_set.txt             # verification set
├── models
│   ├── __init__.py
│   └── svm.py                  # svm model
├── test.py
└── utils
    ├── __init__.py
    ├── data_helper.py          # preprocess util of primer data which like test_set.txt upon 
    └── extract_samples.py      # extracting samples from ACE data
```

## procedures

### extracting samples(optional)

1. extract sentences for ace chinese data set.
2. mark up the sentences in following types:

|0|1|2|3|4|5|6|7|8|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|not any class|life|movement|transaction|business|conflict|contact|personnel|justice|

### segment to words(optional)

if you using chinese data set, you have to using data_helper.py like:
```python
train_text, train_labels, ver_text, ver_labels, test_text, test_labels = data_helper.get_data_and_labels()
``` 

### get stopwords

```python
stopwords = data_helper.get_stopwords()
```

### initial models
```python
# svm
model = SVM(train_text, train_labels, ver_text, ver_labels, test_text, test_labels, stopwords)
```

### train, verification and test
```python
model.train()

model.verification()

model.test()
```

### get result
```python
print('verification accuracy: {:.3}'.format(model.ver_acc))
    
print('test accuracy: {:.3}'.format(model.test_acc))
```

## power by

- ace chinese data set
- [lxml](https://github.com/lxml/lxml)
- [pyltp](https://github.com/HIT-SCIR/pyltp)