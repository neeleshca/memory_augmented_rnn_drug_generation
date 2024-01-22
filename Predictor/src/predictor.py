import numpy as np
import random
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import copy
import os
from pathlib import Path 

PATH_TO_MODEL = "model"
Path(PATH_TO_MODEL).mkdir(exist_ok=True)

l1 = []
l2 = []
PROPERTY_COLUMN = 'ALOGP'

with open(Path("..", "..", "data", "predictor_data.csv")) as file1:
    reader = csv.DictReader(file1, delimiter=',')
    for i in reader:
        l1.append(i['canonical_smile'])
        l2.append((i['canonical_smile'], i[PROPERTY_COLUMN]))


list_120 = []
for i in l2:
    if len(i[0]) <= 120:
        list_120.append(i)

tokens = {'<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n'}
tokens_string = ''.join(sorted(list(tokens)))


size_of_sample = 200000
small_list = random.sample(list_120, size_of_sample)

train, test = train_test_split(small_list, test_size=0.2)
x_train = []
y_train = []
x_test = []
y_test = []
for i in train:
    x_train.append(i[0])
    y_train.append(i[1])
for i in test:
    x_test.append(i[0])
    y_test.append(i[1])


tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tk.fit_on_texts(x_train)

alphabet = tokens_string

char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1
tk.word_index = char_dict.copy()
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

train_sequences = tk.texts_to_sequences(x_train)
test_texts = tk.texts_to_sequences(x_test)

train_data = pad_sequences(train_sequences, maxlen=121, padding='post')
test_data = pad_sequences(test_texts, maxlen=121, padding='post')

train_data = np.array(train_data, dtype='float32')
test_data = np.array(test_data, dtype='float32')


vocab_size = len(tk.word_index)
embedding_weights = [] #(47, 46)
embedding_weights.append(np.zeros(vocab_size)) # first row is pad

for char, i in tk.word_index.items(): # from index 1 to 46
    onehot = np.zeros(vocab_size)
    onehot[i-1] = 1
    embedding_weights.append(onehot)
embedding_weights = np.array(embedding_weights)


input_size = 121
embedding_size = 46


conv_layers = [[256, 7, 3], 
               [256, 7, 3], 
               [256, 3, -1], 
               [256, 3, -1], 
               [256, 3, -1]]

fully_connected_layers = [1024, 1024]
dropout_p = 0.5
optimizer = 'adam'
loss = 'mean_squared_error'

embedding_layer = Embedding(vocab_size+1, 
                            embedding_size,
                            input_length=input_size,
                            weights=[embedding_weights])

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def adjusted_r2_keras(y_true, y_pred):
    r2 = r2_keras(y_true, y_pred)
    numerator = (1-r2)*(size_of_sample-1)
    denominator = size_of_sample-vocab_size-1
    return (1 - (numerator/denominator))


inputs = Input(shape=(input_size,), name='input', dtype='int64')  
x = embedding_layer(inputs)
for filter_num, filter_size, pooling_size in conv_layers:
    x = Conv1D(filter_num, filter_size)(x) 
    x = Activation('relu')(x)
    if pooling_size != -1:
        x = MaxPooling1D(pool_size=pooling_size)(x) 
x = Flatten()(x) 

for dense_size in fully_connected_layers:
    x = Dense(dense_size, activation='relu')(x) 
    x = Dropout(dropout_p)(x)
    
predictions = Dense(1)(x)
model = Model(inputs=inputs, outputs=predictions) 
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) 

model.summary()
checkpoint = ModelCheckpoint('model/model-{epoch:03d}-{loss:03f}-{val_loss:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  


x_train = train_data
x_test = test_data

y_train_final = np.expand_dims(np.array(y_train), axis = 1)
y_test_final = np.expand_dims(np.array(y_test), axis = 1)


model.fit(x_train, y_train_final,
          validation_data=(x_test, y_test_final),
          batch_size=128,
          epochs=100,
          verbose=2,
          callbacks=[checkpoint])
