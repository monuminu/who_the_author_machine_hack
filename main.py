# -*- coding: utf-8 -*-
#importing libraries
import numpy as np
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from text_util import clean_text


def clean_str(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    string = re.sub(r'[^\w]', ' ', string)
    string = re.sub(' +',' ',string)
    return string.strip().lower()

#load the data
train = pd.read_csv('TRAIN.csv')
test  = pd.read_csv('TEST.csv')
data  = pd.concat([train,test], axis = 0).reset_index(drop = True)

#check for missing values
print(train.isnull().sum())
print(test.isnull().sum())

#load text and label
data['text'] = data['text'].map(lambda x: clean_str(x))
data['text'] = data['text'].map(lambda x: clean_str(x))

texts = data['text']
labels = train['author'] 
#static data
EMBEDDING_DIM = 100
MAX_NB_WORDS = 5000
MAX_LENGTH = 100
VALIDATION_SPLIT = 0.2
#tokenize
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
# pad sequences
max_length = max([len(s.split()) for s in texts])
texts = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')
test_texts = texts[18977:]
texts = texts[0:18977]
labels = to_categorical(labels)
no_of_classes = labels.shape[1]
#run word embedding for text 
embeddings_index = {}
f = open('C:/Users/Asus/Desktop/Ipython_notebooks/processed/glove/glove.6B.100d.txt',encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))


embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_LENGTH,
                            trainable=True)


x_train, x_val, y_train, y_val = train_test_split(texts, labels, test_size = VALIDATION_SPLIT)

sequence_input = Input(shape=(MAX_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(50))(embedded_sequences)
preds = Dense(labels.shape[1], activation='softmax')(l_lstm)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

cp=ModelCheckpoint('model_rnn1.hdf5',monitor='val_acc',verbose=1,save_best_only=True)
history=model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=10,callbacks=[cp])

fig2=plt.figure()
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves : RNN',fontsize=16)
fig2.savefig('accuracy_rnn.png')
plt.show()


#Submision
y_pred = model.predict(test_texts)
y_pred_labels = np.argmax(y_pred,axis = 1)
y_pred_df = pd.DataFrame(y_pred_labels, columns = ['author'])
y_pred_df.to_excel('submission2.xlsx',index = False)