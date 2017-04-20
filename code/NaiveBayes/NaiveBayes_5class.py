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
    self.total_awesome_docs = 0
    self.total_good_docs = 0
    self.total_average_docs = 0
    self.total_fair_docs = 0
    self.total_poor_docs = 0

    self.total_awesome_words = 0
    self.total_good_words = 0
    self.total_average_words = 0
    self.total_fair_words = 0
    self.total_poor_words = 0


  def classify(self, words):

    if self.BOOLEAN_NB:
      words = set(words)

    total_docs = self.total_awesome_docs + self.total_good_docs + self.total_average_docs + self.total_fair_docs + self.total_poor_docs
    
    awesome_score = math.log10(float(self.total_awesome_docs)/ total_docs)
    good_score = math.log10(float(self.total_good_docs)/ total_docs)
    average_score = math.log10(float(self.total_average_docs)/ total_docs)
    fair_score = math.log10(float(self.total_fair_docs)/ total_docs)
    poor_score = math.log10(float(self.total_poor_docs)/ total_docs)
    
    vocab_size = len(self.vocab.keys())

# Cal awesome score
    for word in words:
      word_freq = 0
      if word in self.vocab:
        word_freq = self.vocab[word][0]
      temp = float(word_freq + 1) / ( self.total_awesome_words + vocab_size + 1)
      log_temp = math.log10(temp)
      awesome_score += log_temp

# Cal good score
    for word in words:
      word_freq = 0
      if word in self.vocab:
        word_freq = self.vocab[word][1]
      temp = float(word_freq + 1) / ( self.total_good_words + vocab_size + 1)
      log_temp = math.log10(temp)
      good_score += log_temp

# Cal average score
    for word in words:
      word_freq = 0
      if word in self.vocab:
        word_freq = self.vocab[word][2]
      temp = float(word_freq + 1) / ( self.total_average_words + vocab_size + 1)
      log_temp = math.log10(temp)
      average_score += log_temp

# Cal fair score
    for word in words:
      word_freq = 0
      if word in self.vocab:
        word_freq = self.vocab[word][3]
      temp = float(word_freq + 1) / ( self.total_fair_words + vocab_size + 1)
      log_temp = math.log10(temp)
      fair_score += log_temp

# Cal poor score
    for word in words:
      word_freq = 0
      if word in self.vocab:
        word_freq = self.vocab[word][4]
      temp = float(word_freq + 1) / ( self.total_poor_words + vocab_size + 1)
      log_temp = math.log10(temp)
      poor_score += log_temp

    max_score = max(awesome_score,good_score,average_score,fair_score,poor_score)
    if max_score == awesome_score:
      return 'awesome'
    elif max_score == good_score:
      return 'good'
    elif max_score == average_score:
      return 'average'
    elif max_score == fair_score:
      return 'fair'
    else:
      return 'poor'
  

  def addExample(self, klass, words):

    if self.BOOLEAN_NB:
      words = set(words)

    if klass == 'awesome':
      self.total_awesome_docs += 1
    elif klass == 'good':
      self.total_good_docs += 1
    elif klass == 'average':
      self.total_average_docs += 1
    elif klass == 'fair':
      self.total_fair_docs += 1
    else:
      self.total_poor_docs += 1

    for word in words:
      if word not in self.vocab:
        self.vocab[word] = [0,0,0,0,0]

      if klass == 'awesome':
        self.vocab[word][0] += 1
        self.total_awesome_words += 1
      elif klass == 'good':
        self.vocab[word][1] += 1
        self.total_good_words += 1
      elif klass == 'average':
        self.vocab[word][2] += 1
        self.total_average_words += 1
      elif klass == 'fair':
        self.vocab[word][3] += 1
        self.total_fair_words += 1
      else:
        self.vocab[word][4] += 1
        self.total_poor_words += 1

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
    awesomeTrainFileNames = os.listdir('%s/awesome/' % trainDir)
    awesomeTrainFileNames = awesomeTrainFileNames[2500:]

    goodTrainFileNames = os.listdir('%s/good/' % trainDir)
    goodTrainFileNames = goodTrainFileNames[2500:]

    averageTrainFileNames = os.listdir('%s/average/' % trainDir)
    averageTrainFileNames = averageTrainFileNames[2500:]

    fairTrainFileNames = os.listdir('%s/fair/' % trainDir)
    fairTrainFileNames = fairTrainFileNames[2500:]

    poorTrainFileNames = os.listdir('%s/poor/' % trainDir)
    poorTrainFileNames = poorTrainFileNames[2500:]

    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in awesomeTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/awesome/%s' % (trainDir, fileName))
        example.klass = 'awesome'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in goodTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/good/%s' % (trainDir, fileName))
        example.klass = 'good'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in averageTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/average/%s' % (trainDir, fileName))
        example.klass = 'average'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in fairTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/fair/%s' % (trainDir, fileName))
        example.klass = 'fair'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in poorTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/poor/%s' % (trainDir, fileName))
        example.klass = 'poor'
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
    print 'usage:\NaiveBayes_5class.py <train_dir>'
    sys.exit(0)
  main(sys.argv[1])
