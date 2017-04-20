import sys
import getopt
import os
import math
import operator
import nltk
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class NaiveBayes:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:

    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """NaiveBayes initialization"""
    self.BOOLEAN_NB = True
    self.numFolds = 10
    self.vocab = {}
    self.total_pos_docs = 0
    self.total_neg_docs = 0
    self.total_avg_docs = 0
    self.total_pos_words = 0
    self.total_neg_words = 0
    self.total_avg_words = 0


  def classify(self, words):

    if self.BOOLEAN_NB:
      words = set(words)

    pos_score = math.log10(float(self.total_pos_docs)/ (self.total_pos_docs + self.total_neg_docs + self.total_avg_docs))
    avg_score = math.log10(float(self.total_avg_docs)/ (self.total_pos_docs + self.total_neg_docs + self.total_avg_docs))
    neg_score = math.log10(float(self.total_neg_docs)/ (self.total_pos_docs + self.total_neg_docs + self.total_avg_docs))
    vocab_size = len(self.vocab.keys())

# Cal positive score
    for word in words:
      word_freq = 0
      if word in self.vocab:
        word_freq = self.vocab[word][0]
      temp = float(word_freq + 1) / ( self.total_pos_words + vocab_size + 1)
      log_temp = math.log10(temp)
      pos_score += log_temp

# Cal average score
    for word in words:
      word_freq = 0
      if word in self.vocab:
        word_freq = self.vocab[word][1]
      temp = float(word_freq + 1) / ( self.total_avg_words + vocab_size + 1)
      log_temp = math.log10(temp)
      avg_score += log_temp

# Cal negative score
    for word in words:
      word_freq = 0
      if word in self.vocab:
        word_freq = self.vocab[word][2]
      temp = float(word_freq + 1) / ( self.total_neg_words + vocab_size + 1)
      log_temp = math.log10(temp)
      neg_score += log_temp

    max_score = max(pos_score,avg_score,neg_score)
    if max_score == pos_score:
      return 'pos'
    elif max_score == avg_score:
      return 'avg'
    else:
      return 'neg'
  

  def addExample(self, klass, words):

    if self.BOOLEAN_NB:
      words = set(words)

    if klass == 'pos':
      self.total_pos_docs += 1
    elif klass == 'avg':
      self.total_avg_docs += 1
    else:
      self.total_neg_docs += 1

    for word in words:
      if word not in self.vocab:
        self.vocab[word] = [0,0,0]

      if klass == 'pos':
        self.vocab[word][0] += 1
        self.total_pos_words += 1
      elif klass == 'avg':
        self.vocab[word][1] += 1
        self.total_avg_words += 1
      else:
        self.vocab[word][2] += 1
        self.total_neg_words += 1

    return
  
  
  def readFile(self, fileName):

    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line.lower())
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):

    bigrams = []
    for item in nltk.bigrams (re.split('\W+', s)): bigrams.append(' '.join(item))
    return bigrams + re.split('\W+', s)


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    posTrainFileNames = posTrainFileNames[2500:]

    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    negTrainFileNames = negTrainFileNames[2500:]

    avgTrainFileNames = os.listdir('%s/avg/' % trainDir)
    avgTrainFileNames = avgTrainFileNames[2500:]
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in avgTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/avg/%s' % (trainDir, fileName))
        example.klass = 'avg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits  

def test10Fold(trainDir):
  nb = NaiveBayes()
  splits = nb.crossValidationSplits(trainDir)
  avgAccuracy = 0.0
  fold = 0
  guesses = []
  klasses = []
  for split in splits:
    classifier = NaiveBayes()
    accuracy = 0.0
    for example in split.train:
      words = example.words
      classifier.addExample(example.klass, words)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if fold==0:
        guesses.append(guess)
        klasses.append(example.klass)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 

    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy
  print('Confusion Matrix : ')
  print confusion_matrix(klasses, guesses)
  print('Classification Report :')
  print classification_report(klasses, guesses) 

def main(trainDir):
  test10Fold(trainDir)

if __name__ == "__main__":
  if (len(sys.argv) != 2):
    print 'usage:\NaiveBayes_3class.py <train_dir>'
    sys.exit(0)
  main(sys.argv[1])
