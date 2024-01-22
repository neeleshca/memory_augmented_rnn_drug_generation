import tensorflow as tf

import argparse
parser = argparse.ArgumentParser()
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
from stackRNN import stackRNNCell

# Taken from the release data generator file. 
# Length of data is 120. 
# Including start and end token, it's 122. Input and target are one less than that. 
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
        # https://stackoverflow.com/questions/42440565/how-to-feed-back-rnn-output-to-input-in-tensorflow
        global initial_state_placeholder
        # Batch size
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
    def __init__(self, inputs, outputs):
        super(BuildTModel, self).__init__(inputs)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = outputs, logits = self.output_logits_decoded)
        self.individual_loss = cross_entropy
        self.loss = tf.reduce_sum(cross_entropy)/batch_size
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        trainable_variables = tf.trainable_variables()
        print(trainable_variables)
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


with tf.variable_scope('root'):
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, None))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
    model = BuildTModel(inputs_placeholder, labels_placeholder)
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
    # print(inp_result, target_result)
    return (inp_result, target_result)

def get_predicted_character(data, output_logits):
    output_logits = np.squeeze(output_logits)
    prob_output = sess.run(prob_output_softmax, feed_dict={predicted_char_placeholder:output_logits})
    prob_output = prob_output.astype('float64')
    prob_output /= prob_output.sum()
    next_char = np.random.multinomial(1, prob_output, size = 1)
    char_index = np.where(next_char[0]==1.0)
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
        pred_char = get_predicted_character(data, output_logits[i])
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
            pred_char = get_predicted_character(data, output_logits[j])
            new_sample[j][i]=pred_char
            inp[j] = data.char_tensor(pred_char)
    return new_sample

def write_now(filep, msg):
    """Write msg to the file given by filep, forcing the msg to be written to the filesystem immediately (now).

    Without this, if you write to files, and then execute programs
    that should read them, the files will not show up in the program
    on disk.
    """
    filep.write(msg)
    filep.flush()
    # The above call to flush is not enough to write it to disk *now*;
    # according to https://stackoverflow.com/a/41506739/257924 we must
    # also call fsync:
    os.fsync(filep)

def generate_strings(path, num_strings):
    generated_file = open(path, "w")
    for num in range(num_strings):
        sample = evaluate(gen_data)
        l = []
        for i1 in sample:
            tmp_str = ""
            for j in i1:
                tmp_str += j.decode()
            l.append(tmp_str)

        for i2 in l:
            generated_file.write(i2)
            generated_file.write("\n")        
    generated_file.close()

print("FINISHED INITIALIZATION\n")

starting_model_number = 0
num_train_steps = 10
num_save = 6
load = 1
train = 0
generate = 1
num_evaluate = 5000 # Call evaluate after every num_evaluate iterations
num_strings = 50 # number of strings to generate every num_evaluate iterations

if(load):
    saver.restore(sess, f"./{PATH_TO_MODEL}/model.ckpt")

losses_file  = open("losses.txt", "a")
losses_backup_file  = open("losses_backup.txt", "a")
op_losses = 100000000

tf.get_default_graph().finalize()

start_time = time.time()
if(generate == 1):
    generate_strings(f'{PATH_TO_GENERATED_FILE}/generated_file_max.txt', num_strings)


if(train == 1):
    for i in range(starting_model_number+1, num_train_steps+starting_model_number+1):
        full_inputs = np.empty(shape = (batch_size, max_string_length))
        decoded_output = np.empty(shape = (batch_size, max_string_length))
        for j in range(batch_size):
            inp, target = gen_data.random_training_set(None)
            inp, target = pad_input_and_target(inp, target, max_string_length)
            full_inputs[j] = add_axis_0(inp)
            decoded_output[j] = add_axis_0(target)
        decoded_output = decoded_output.astype(int)
        train_loss, _= sess.run([model.loss, model.train_op],
            feed_dict={
                inputs_placeholder: full_inputs,
                labels_placeholder: decoded_output
            })
        # print(f'Train loss ({i}): {train_loss/seq_len}\n')
        if(i%10==0):
            write_now(losses_file, f'Train loss ({i}): {train_loss/seq_len}\n')
            write_now(losses_backup_file, f'Train loss ({i}): {train_loss/seq_len}\n')
            print(f'Train loss ({i}): {train_loss/seq_len}\n')
            if(i%100==0):
                write_now(losses_file, f'Time for iteration ({i}): {time.time() - start_time}\n')
                write_now(losses_backup_file, 
                f'Time for iteration ({i}): {time.time() - start_time}\n')
                
        if(i%num_evaluate==0):
            generate_strings(f'{PATH_TO_GENERATED_FILE}/generated_file_{i}.txt', num_strings)
    save_path = saver.save(sess, f'./{PATH_TO_MODEL}/train_step_{i}/model.ckpt')
    
end_time = time.time()
print(end_time - start_time)   
