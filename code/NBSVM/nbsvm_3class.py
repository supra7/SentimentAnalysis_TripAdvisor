import glob
import os
import string
import random
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from nbsvm import NBSVM

def prep_Vectorizer():
    print("Preparing Vectorizer")

    train_pos = glob.glob(os.path.join('../../data', 'pos', '*.txt'))[:4500]
    train_avg = glob.glob(os.path.join('../../data', 'avg', '*.txt'))[:4500]
    train_neg = glob.glob(os.path.join('../../data', 'neg', '*.txt'))[:4500]

    token_pattern = r'\w+|[%s]' % string.punctuation

    vectorizer = TfidfVectorizer('filename', ngram_range=(1, 3),
                                 token_pattern=token_pattern,
                                 binary=True,
                                 sublinear_tf=True
                                 )

    vectorizer.fit(train_pos+train_avg+train_neg)

    print("Vocabulary Size: %s" % len(vectorizer.vocabulary_))
    
    return vectorizer

def load_trainSet(vectorizer,class1,class2):
    print("Vectorizing Training Text")

    train_pos = glob.glob(os.path.join('../../data',  class1, '*.txt'))[:4500]
    train_neg = glob.glob(os.path.join('../../data',  class2, '*.txt'))[:4500]

    X_train = vectorizer.transform(train_pos+train_neg)
    y_train = np.array([1]*len(train_pos)+[0]*len(train_neg))

    return X_train, y_train

def load_testSet(vectorizer):
    print("Vectorizing Testing Text")

    test_pos = glob.glob(os.path.join('../../data',  'pos', '*.txt'))[4500:]
    test_avg = glob.glob(os.path.join('../../data',  'avg', '*.txt'))[4500:]
    test_neg = glob.glob(os.path.join('../../data',  'neg', '*.txt'))[4500:]

    X_test = vectorizer.transform(test_pos + test_avg+ test_neg)
    y_test = np.array(["pos"]*len(test_pos) + ["avg"]*len(test_avg) + ["neg"]*len(test_neg))

    return X_test, y_test

def main():
    np.set_printoptions(threshold=np.inf)

    vectorizer = prep_Vectorizer()
    X_test, y_test = load_testSet(vectorizer)
    mnbsvm = NBSVM()

    scores=[[0,0,0] for i in range(len(y_test))]
    result = np.array(["none"]*len(y_test))
    print("Fitting Models now")

    X_train, y_train = load_trainSet(vectorizer,'pos','avg')
    print('Fitting pos-avg classifier')
    mnbsvm.fit(X_train, y_train)
    pos_avg_res = mnbsvm.predict(X_test)
    # print pos_avg_res

    X_train, y_train = load_trainSet(vectorizer,'avg','neg')
    print('Fitting avg-neg classifier')
    mnbsvm.fit(X_train, y_train)
    avg_neg_res = mnbsvm.predict(X_test)
    # print avg_neg_res

    X_train, y_train = load_trainSet(vectorizer,'pos','neg')
    print('Fitting pos-neg classifier')
    mnbsvm.fit(X_train, y_train)
    pos_neg_res = mnbsvm.predict(X_test)
    # print pos_neg_res

    random_count = 0
    for i in range(len(y_test)):
        if pos_avg_res[i] == 1:
            scores[i][0] += 1
        else:
            scores[i][1] += 1

        if avg_neg_res[i] == 1:
            scores[i][1] += 1
        else:
            scores[i][2] += 1

        if pos_neg_res[i] == 1:
            scores[i][0] += 1
        else:
            scores[i][2] += 1

        if scores[i][0] == scores[i][1] == scores[i][2]:
            result[i] = random.choice(["pos","avg","neg"])
            random_count += 1
        else:
            max_value = max(scores[i])
            max_index = scores[i].index(max_value)
            if max_index == 0:
                result[i] = "pos"
            elif max_index == 1:
                result[i] = "avg"
            else:
                result[i] = "neg"


    print('Test Accuracy: %s' % accuracy_score(y_test, result))
    print('Confusion Matrix : ')
    print confusion_matrix(y_test, result,labels=["pos", "avg", "neg"])
    print('Classification Report :')
    print classification_report(y_test, result,labels=["pos", "avg", "neg"])

if __name__ == '__main__':
    main()
