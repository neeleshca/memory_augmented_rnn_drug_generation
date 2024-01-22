import tensorflow as tf


from stackRNN import stackRNNCell
import sys
sys.path.append('../../../release/')
from data import GeneratorData
from tensorflow.keras import layers
import numpy as np   
import time
import random
import os
import subprocess
from pathlib import Path
from rdkit import Chem 

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
from keras import backend as K

max_string_length = 121 
seq_len = max_string_length

embedding_length = 1024
num_layers = 1
num_controller_layers = num_layers 
num_units = 1024
num_controller_units = num_units
stack_width = 1024
stack_depth = 200

learning_rate = 0.001
max_grad_norm = 50
batch_size = 26

num_bits_per_vector = embedding_length

PATH_TO_GENERATED_FILE = "generated_files"
Path(PATH_TO_GENERATED_FILE).mkdir(exist_ok=True)

PATH_TO_MODEL = "model"
Path(PATH_TO_MODEL).mkdir(exist_ok=True)

PREDICTOR_MODEL_PATH = '../../../models/predictor/alogp.h5'

def get_predicted_value(list_input, batch_size):
    with tf.Session() as sess:
        tokens = {'<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n'}
        tokens_string = ''.join(sorted(list(tokens)))
        tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        tk.fit_on_texts(list_input)
        alphabet = tokens_string
        char_dict = {}
        for i, char in enumerate(alphabet):
            char_dict[char] = i + 1
        tk.word_index = char_dict.copy()
        tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
        train_sequences = tk.texts_to_sequences(list_input)
        train_data = pad_sequences(train_sequences, maxlen=121, padding='post')
        train_data = np.array(train_data, dtype='float32')
        predictor_model = load_model(PREDICTOR_MODEL_PATH, compile = False)
        return predictor_model.predict(train_data, batch_size=batch_size) 
        

gen_data_path = '../../../data/chembl_22_clean_1576904_sorted_std_final.smi'

tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', 
                         cols_to_read=[0], keep_header=True, tokens=tokens)

char_size = len(tokens)

initial_state_placeholder = None
class BuildModel(object):
    def __init__(self, inputs):
        self.embedding_layer = layers.Embedding(char_size, embedding_length)
        self.inputs = self.embedding_layer(inputs)
        self._build_model()
    def _build_model(self):
        cell = stackRNNCell(num_layers = num_layers, num_units = num_units, stack_width = stack_width, stack_depth = stack_depth )
        global initial_state_placeholder
        initial_state_placeholder = cell.zero_state(batch_size, tf.float32)
        output_sequence, final_state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=self.inputs,
            time_major=False,
            dtype=tf.float32,
            initial_state=initial_state_placeholder)

        self.state_final = final_state 
        self.output_logits = output_sequence
        self.output_logits_decoded = tf.layers.dense(self.output_logits,char_size)

class BuildTModel(BuildModel):
    def __init__(self, inputs, outputs, predicted):
        super(BuildTModel, self).__init__(inputs)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = outputs, logits = self.output_logits_decoded)
        self.individual_loss = cross_entropy
        self.loss = tf.reduce_sum(cross_entropy)/batch_size
        self.predicted_loss = (tf.math.exp((tf.reduce_sum(predicted)/batch_size)/1.5))
        self.total_loss = self.loss + self.predicted_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.total_loss, trainable_variables), max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


with tf.variable_scope('root'):
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, None))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
    predicted_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    model = BuildTModel(inputs_placeholder, labels_placeholder, predicted_placeholder)
    predicted_char_placeholder = tf.placeholder(tf.float32, shape=(char_size))
    prob_output_softmax = tf.nn.softmax(predicted_char_placeholder)


initializer = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=None)
sess = tf.Session()
sess.run(initializer)

def add_axis_0(inp):
    return np.expand_dims(inp, axis = 0)

def pad_input_and_target(inp, target, max_string_length):
    inp_result = np.ones(max_string_length, dtype=int) 
    target_result = np.ones(max_string_length, dtype=int) 
    inp_result[0:len(inp)] = inp
    target_result[0:len(target)] = target
    return (inp_result, target_result)

def get_predicted_character(data, output_logits, epsilon = None):
    output_logits = np.squeeze(output_logits)
    prob_output = sess.run(prob_output_softmax, feed_dict={predicted_char_placeholder:output_logits})
    prob_output = prob_output.astype('float64')
    prob_output /= prob_output.sum()
    next_char = np.random.multinomial(1, prob_output, size = 1)
    char_index = np.where(next_char[0]==1.0)
    if(epsilon is not None):
        if(random.uniform(0, 1) <= epsilon):
            next_char = np.random.multinomial(2, prob_output, size = 1)
            char_index = np.where(next_char[0]==1.0)
            if(len(char_index[0]) == 0):
                char_index = np.where(next_char[0]==2.0)        
    np.random.shuffle(char_index[0])
    pred_char = data.all_characters[char_index[0][0]]
    return pred_char
    
def evaluate(data, start_char = '<', end_char = '>', predict_length = 100):
    full_inputs = np.empty(shape = (batch_size, 1))    
    new_sample = np.empty(shape = (batch_size, predict_length), dtype = np.dtype('bytes'))
    inp = np.empty(shape = (batch_size, 1))    
    for i in range(batch_size):
        starting_char = data.char_tensor(start_char)
        full_inputs[i] = add_axis_0(starting_char)
        new_sample[i][0] = start_char

    output_logits, next_state = sess.run([model.output_logits_decoded, model.state_final],
    feed_dict={
        inputs_placeholder: full_inputs,
    })
    for i in range(batch_size):
        pred_char = get_predicted_character(data, output_logits[i], 0.7)
        new_sample[i][1]=pred_char
        inp[i] = data.char_tensor(pred_char)

    for i in range(2, predict_length):
        for j in range(batch_size):
            full_inputs[j] = add_axis_0(inp[j])

        output_logits, next_state = sess.run([model.output_logits_decoded, model.state_final],
        feed_dict={
            inputs_placeholder: full_inputs,
            initial_state_placeholder.hidden_state: next_state.hidden_state,
            initial_state_placeholder.stack: next_state.stack
        })
        for j in range(batch_size):
            pred_char = get_predicted_character(data, output_logits[j], 0.7)
            new_sample[j][i]=pred_char
            inp[j] = data.char_tensor(pred_char)
    return new_sample

def get_strings():
    list1 = []
    for num in range(2):
        sample = evaluate(gen_data)
        l = []
        for i in sample:
            tmp_str = ""
            for j in i:
                tmp_str += j.decode()
            l.append(tmp_str)
        for i in l:
            if(i.find('>') != -1):
                list1.append(i[1:i.find('>')])
    valid_list = []
    for i in list1:
        m = Chem.MolFromSmiles(i)
        if m is not None:
            valid_list.append(i)
    return random.sample(valid_list, batch_size)
    
starting_model_number = 0
num_train_steps = 100
load = 1

if(load):
    saver.restore(sess, f"./model/model.ckpt")

start_time = time.time()


for i in range(starting_model_number+1, num_train_steps+starting_model_number+1):
    generated_strings = get_strings()
    predicted_val = get_predicted_value(generated_strings, len(generated_strings))
    full_inputs = np.empty(shape = (batch_size, max_string_length))
    decoded_output = np.empty(shape = (batch_size, max_string_length))
    predicted_array = np.empty(shape = (batch_size))
    loss_list = []
    predicted_loss_list = []
    for j in range(batch_size):
        inp, target = gen_data.random_training_set(None)
        inp, target = pad_input_and_target(inp, target, max_string_length)
        full_inputs[j] = add_axis_0(inp)
        decoded_output[j] = add_axis_0(target)
        predicted_array[j] = predicted_val[j]

    decoded_output = decoded_output.astype(int)
    train_loss, _= sess.run([model.loss, model.train_op],
        feed_dict={
            inputs_placeholder: full_inputs,
            labels_placeholder: decoded_output,
            predicted_placeholder: predicted_array
        })
    print(f'Train loss ({i}): {train_loss/seq_len}\n')
    loss_list.append(train_loss)
    predicted_loss_list.append(np.average(predicted_array))
    if(len(loss_list) > 5):
        flag = 1
        for index in range(len(loss_list) - 1, len(loss_list) - 6, -1):
            if(predicted_loss_list[index] > 0.7):
                flag = 0 
        if(flag == 1):
            save_path = saver.save(sess, f'./{PATH_TO_MODEL}/train_step_{i}/model.ckpt')
            break
    
end_time = time.time()
print(end_time - start_time)