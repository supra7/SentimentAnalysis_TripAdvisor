
# coding: utf-8

# In[1]:

import sys
import tensorflow as tf
import json
import os
n_epoch = 3
import pandas as pd


# In[2]:

import csv
from pandas.io.json import json_normalize
dirs = os.listdir("Downloads/json/")
print len(dirs)
with open('tripadv.csv','wb') as f:
    writer = csv.writer(f)
    writer.writerow(["Ratings","Content"])
    for filename in dirs:
        with open('Downloads/json/'+filename) as fi:
            
            try:
                d = json.load(fi)
            except:
                d={}
                d["Reviews"] = []
        for x in d["Reviews"]:
            if float(x["Ratings"]["Overall"]) in [1.0,2.0,3.0,4.0,5.0]:
                writer.writerow([float(x["Ratings"]["Overall"].encode('utf-8')),x["Content"].encode('utf-8')])


# In[3]:

import pandas as pd
df = pd.read_csv('out.csv')
df.head(10)


# In[4]:

X = df.Content
y = df.Ratings
X.head(10)
y.head(10)


# In[5]:

def get_positive_or_negative_classes_two_classes(y):
    return y.apply(lambda val: 0 if val<=2 else 1)
def get_positive_or_negative_classes_three_classes(y):
    return y.apply(lambda val: 0 if val<=2 else 2 if val > 3 else 1)
def get_positive_or_negative_classes_five_classes(y):
    return y.apply(lambda val: 0 if val=='' else val)


# In[6]:

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# Comment for getting 5 classes
#y_train = get_positive_or_negative_classes(y_train)
#y_test = get_positive_or_negative_classes(y_test)
#y_train = get_positive_or_negative_classes_five_classes(y_train)
#y_test = get_positive_or_negative_classes_five_classes(y_test)


# In[7]:

import string
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(1,1), token_pattern=r'\b\w{1,}\b')


# In[8]:

vect.fit(X_train)
vocab = vect.vocabulary_
len(vocab)


# In[9]:

def convert_X_to_X_word_ids(X):
    return X.apply( lambda x: [vocab[w] for w in [w.lower().strip() for w in x.split()] if w in vocab] )


# In[10]:

X_train_word_ids = convert_X_to_X_word_ids(X_train)
X_test_word_ids  = convert_X_to_X_word_ids(X_test)


# In[66]:

X_train_word_ids.head()


# In[67]:

X_train.head()


# In[11]:

X_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=20, value=0)
X_test_padded_seqs  = pad_sequences(X_test_word_ids , maxlen=20, value=0)


# In[69]:

print('X_train_padded_seqs.shape:', X_train_padded_seqs.shape)
print('X_test_padded_seqs.shape:', X_test_padded_seqs.shape)


# In[11]:

pd.DataFrame(X_train_padded_seqs).head()


# In[12]:

pd.DataFrame(X_test_padded_seqs).head()


# In[12]:

unique_y_labels = list(y_train.value_counts().index)
unique_y_labels


# In[13]:

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(unique_y_labels)


# In[14]:

y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), nb_classes=len(unique_y_labels))
y_test  = to_categorical(y_test.map(lambda x:  le.transform([x])[0]), nb_classes=len(unique_y_labels))


# In[41]:

y_train


# In[89]:

print('y_train.shape:', y_train.shape)
print('y_test.shape:', y_test.shape)


# In[15]:

size_of_each_vector = X_train_padded_seqs.shape[1]
vocab_size = len(vocab)
no_of_unique_y_labels = len(unique_y_labels)


# In[91]:

print('size_of_each_vector:', size_of_each_vector)
print('vocab_size:', vocab_size)
print('no_of_unique_y_labels:', no_of_unique_y_labels)


# In[16]:

net = tflearn.input_data([None, size_of_each_vector]) # The first element is the "batch size" which we set to "None"
net = tflearn.embedding(net, input_dim=vocab_size, output_dim=128) # input_dim: vocabulary size
net = tflearn.lstm(net, 128, dropout=0.6) # Set the dropout to 0.6
net = tflearn.fully_connected(net, no_of_unique_y_labels, activation='softmax') # relu or softmax
net = tflearn.regression(net, 
                         optimizer='adam',  # adam or ada or adagrad # sgd
                         learning_rate=1e-4,
                         loss='categorical_crossentropy')


# In[17]:

model = tflearn.DNN(net, tensorboard_verbose=0)


# In[18]:

model.fit(X_train_padded_seqs, y_train, 
          validation_set=(X_test_padded_seqs, y_test), 
          n_epoch=n_epoch,
          show_metric=True, 
          batch_size=100)


# In[21]:

import numpy as np
from sklearn import metrics


# In[24]:

pred_classes = [np.argmax(i) for i in model.predict(X_test_padded_seqs)]
true_classes = [np.argmax(i) for i in y_test]

print('\nRNN Classifier\'s Accuracy: %0.5f\n' % metrics.accuracy_score(true_classes, pred_classes))



# In[30]:

# print le.inverse_transform(pred_classes)
print ('Confusion Matrix:', metrics.confusion_matrix(le.inverse_transform(true_classes), le.inverse_transform(pred_classes)))
print ('Classification Report')
print metrics.classification_report(le.inverse_transform(true_classes), le.inverse_transform(pred_classes))


# In[20]:

ids_of_titles = range(0,100) # range(X_test.shape[0]) 

for i in ids_of_titles:
    pred_class = np.argmax(model.predict([X_test_padded_seqs[i]]))
    true_class = np.argmax(y_test[i])
    
    print(X_test.values[i])
    print('pred_class:', le.inverse_transform(pred_class))
    print('true_class:', le.inverse_transform(true_class))
    print('')


# In[81]:

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
seed = 7
numpy.random.seed(seed)


# In[30]:

def get_positive_or_negative_classes(y):
    return y.apply(lambda val: 0 if val<=2 else 2 if val > 3 else 0 )


# In[88]:

# np_array = np.genfromtxt ('out.csv', delimiter=",")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
max_words = 20
X_train_word_ids = convert_X_to_X_word_ids(X_train)
X_test_word_ids  = convert_X_to_X_word_ids(X_test)
X_train_word_ids = np.array(X_train_word_ids)
X_test_word_ids = np.array(X_test_word_ids)
y_train = get_positive_or_negative_classes_two_classes(y_train)
y_test = get_positive_or_negative_classes_two_classes(y_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_train_padded_seqs = sequence.pad_sequences(X_train_word_ids, maxlen=max_words, value=0)
X_test_padded_seqs  = sequence.pad_sequences(X_test_word_ids , maxlen=max_words, value=0)


# In[ ]:




# In[92]:

model = Sequential()
top_words = len(vocab)
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[85]:

model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[93]:

model.fit(X_train_padded_seqs, y_train, validation_data=(X_test_padded_seqs, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test_padded_seqs, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:




# In[ ]:




# In[50]:




# In[51]:




# In[ ]:




# In[ ]:




# In[ ]:



