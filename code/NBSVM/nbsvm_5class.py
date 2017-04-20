import glob
import os
import string
import random
import numpy as np
import re
from collections import defaultdict
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from nbsvm import NBSVM

def prep_Vectorizer():
    print("Preparing Vectorizer")

    train_awesome = glob.glob(os.path.join('../../data', 'awesome', '*.txt'))[:4500]
    train_good = glob.glob(os.path.join('../../data',  'good', '*.txt'))[:4500]
    train_average = glob.glob(os.path.join('../../data',  'average', '*.txt'))[:4500]
    train_fair = glob.glob(os.path.join('../../data',  'fair', '*.txt'))[:4500]
    train_poor = glob.glob(os.path.join('../../data',  'poor', '*.txt'))[:4500]

    token_pattern = r'\w+|[%s]' % string.punctuation

    vectorizer = TfidfVectorizer('filename', ngram_range=(1, 2),
                                 token_pattern=token_pattern,
                                 binary=True,
                                 sublinear_tf=True)
    vectorizer.fit(train_awesome+train_good+train_average+train_fair+train_poor)

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

    test_awesome = glob.glob(os.path.join('../../data', 'awesome', '*.txt'))[4500:]
    test_good = glob.glob(os.path.join('../../data',  'good', '*.txt'))[4500:]
    test_average = glob.glob(os.path.join('../../data',  'average', '*.txt'))[4500:]
    test_fair = glob.glob(os.path.join('../../data',  'fair', '*.txt'))[4500:]
    test_poor = glob.glob(os.path.join('../../data',  'poor', '*.txt'))[4500:]

    X_test = vectorizer.transform(test_awesome+test_good+test_average+test_fair+test_poor)
    y_test = np.array([5]*len(test_awesome) + [4]*len(test_good) + [3]*len(test_average) + [2]*len(test_fair) + [1]*len(test_poor))

    return X_test, y_test

def main():
    np.set_printoptions(threshold=np.inf)

    vectorizer = prep_Vectorizer()
    X_test, y_test = load_testSet(vectorizer)
    mnbsvm = NBSVM()

    scores=[defaultdict(int) for i in range(len(y_test))]
    result = np.array([0]*len(y_test))
    print 'length of result : ' + str(len(result))
    print("Fitting Models now")

    classes = ['awesome','good','average','fair','poor']
    predictions = {}

    for combo in itertools.combinations(classes, 2): 
        X_train, y_train = load_trainSet(vectorizer,combo[0],combo[1])
        print('Fitting classifier : ' + " ".join(combo))
        mnbsvm.fit(X_train, y_train)
        predictions[" ".join(combo)] = mnbsvm.predict(X_test)

    random_count = 0
    for i in range(len(y_test)):
        for combo in predictions:
            [class1,class2] = combo.split()
            if predictions[combo][i] == 1:
                scores[i][class1] += 1
            else:
                scores[i][class2] += 1

        max_value = max(scores[i].values())
        result_classes = []
        for klass in scores[i]:
            if scores[i][klass] == max_value:
                result_classes.append(klass)

        result_class = random.choice(result_classes)
        if result_class == 'awesome':
            result[i] = 5
        elif result_class == 'good':
            result[i] = 4
        elif result_class == 'average':
            result[i] = 3
        elif result_class == 'fair':
            result[i] = 2
        else:
            result[i] = 1


    print('Test Accuracy: %s' % accuracy_score(y_test, result))
    print('Confusion Matrix : ')
    print confusion_matrix(y_test, result)
    print('Classification Report :')
    print classification_report(y_test, result)


if __name__ == '__main__':
    main()
