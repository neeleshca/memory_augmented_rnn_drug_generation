from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
from keras import backend as K

import tensorflow as tf

import os, sys

sys.path.append('../../release')

from data import GeneratorData
from tensorflow.keras import layers
import numpy as np   
import random
import os
import subprocess
from rdkit import Chem
import random
from enum import Enum



max_string_length = 121 
seq_len = max_string_length

embedding_length = 1024
num_layers = 1
num_controller_layers = num_layers 
num_units = 1024
num_controller_units = num_units
memory_size = 1024
clip_value = 20
clip_controller_output_to_value = clip_value
init_mode = 'constant'
learning_rate = 0.001
max_grad_norm = 50
batch_size = 10
num_bits_per_vector = embedding_length

class GeneratorEnum(Enum):
    simple = 1
    gru = 2
    lstm = 3

class BiasEnum(Enum):
    maximization = 1
    minimization = 2

class PropertyEnum(Enum):
    logP = 1
    Benzene = 2

# order of command line options = Gen bias Prop

# Value initialization section

rnn_option_str = sys.argv[1]
bias_option_str = sys.argv[2]
property_option_str = sys.argv[3]
run_number = int(sys.argv[4])

if (bias_option_str == "max"):
    bias_option = BiasEnum.maximization
elif (bias_option_str == "min"):
    bias_option = BiasEnum.minimization

if (property_option_str == "Benzene"):
    property_option = PropertyEnum.Benzene
elif (property_option_str == "logP"):
    property_option = PropertyEnum.logP

if (rnn_option_str == "lstm"):
    rnn_option = GeneratorEnum.lstm
elif (rnn_option_str == "gru"):
    rnn_option = GeneratorEnum.gru
elif (rnn_option_str == "simple"):
    rnn_option = GeneratorEnum.simple

output_run_path = "generated_strings/"


completed_iterations = 0

if (bias_option == BiasEnum.maximization):
    if (property_option == PropertyEnum.logP):
        early_stop_value = 3.8
    elif (property_option == PropertyEnum.Benzene):
        early_stop_value = 1.65
elif (bias_option == BiasEnum.minimization):
    if (property_option == PropertyEnum.logP):
        early_stop_value = 1.8
    elif (property_option == PropertyEnum.Benzene):
        early_stop_value = 0.7

if not os.path.exists(output_run_path):
    os.makedirs(output_run_path)

gen_data_path = '../../Data/chembl_22_clean_1576904_sorted_std_final.smi'

tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', 
                         cols_to_read=[0], keep_header=True, tokens=tokens)

char_size = len(tokens)

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def fun123(list_input, batch_size):
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
        print("Loading Predictor Model")
        if (property_option == PropertyEnum.Benzene):
            predictor_model = load_model('models/predictor/benzene.h5', compile = False)
        elif (property_option == PropertyEnum.logP):
            predictor_model = load_model('models/predictor/alogp.h5', compile = False)
        print("Load successful")
        return predictor_model.predict(train_data, batch_size=batch_size) 

#tf.reset_default_graph()

initial_state_placeholder = None
class BuildModel(object):
    def __init__(self, inputs):
        self.embedding_layer = layers.Embedding(char_size, embedding_length)
        self.inputs = self.embedding_layer(inputs)
        self._build_model()
    def _build_model(self):
        def single_cell(num_units):
            if (rnn_option == GeneratorEnum.simple):
                return tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units)
            elif (rnn_option == GeneratorEnum.gru):
                return tf.contrib.rnn.GRUCell(num_units)    
            elif (rnn_option == GeneratorEnum.lstm):
                return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias = 1.0)

        rnn_layers = [single_cell(num_units) for _ in range(num_layers)]
        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_layers)

        # https://stackoverflow.com/questions/42440565/how-to-feed-back-rnn-output-to-input-in-tensorflow
        global initial_state_placeholder
        # Batch size
        initial_state_placeholder = multi_rnn_cell.zero_state(batch_size, tf.float32)
        output_sequence, final_state = tf.nn.dynamic_rnn(
            cell=multi_rnn_cell,
            inputs=self.inputs,
            time_major=False,
            dtype=tf.float32,
            initial_state=initial_state_placeholder)

        self.state_final = final_state 
        self.output_logits = output_sequence
        self.output_logits_decoded = tf.layers.dense(self.output_logits,char_size)

class BuildTModel(BuildModel):
    def __init__(self, inputs, outputs, predicted, repeated):
        super(BuildTModel, self).__init__(inputs)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = outputs, logits = self.output_logits_decoded)
        self.individual_loss = cross_entropy
        self.normal_loss = tf.reduce_sum(cross_entropy)/batch_size
        self.loss = tf.reduce_sum(cross_entropy)/batch_size + tf.reduce_sum(predicted)/batch_size

        if (bias_option == BiasEnum.maximization):
            if (property_option == PropertyEnum.logP):
                self.predicted_loss = -1*(tf.math.exp((tf.reduce_sum(predicted)/batch_size)/1.5)) # Maximization logP       
            elif (property_option == PropertyEnum.Benzene):
                self.predicted_loss = -1*(tf.math.exp((tf.reduce_sum(predicted)/batch_size)*1.8)) # Maximization benzene
        elif (bias_option == BiasEnum.minimization):
            if (property_option == PropertyEnum.logP):
                self.predicted_loss = 4*(tf.math.exp(tf.reduce_sum(predicted)/batch_size)) # Minimization logP
            elif (property_option == PropertyEnum.Benzene):
                self.predicted_loss = 1*(tf.math.exp((tf.reduce_sum(predicted)/batch_size)*2))  # Minimization benzene      
        
        self.total_loss = self.loss + self.predicted_loss + repeated
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.total_loss, trainable_variables), max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

with tf.variable_scope('root'):
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, None))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, None))
    predicted_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    repeated_placeholder = tf.placeholder(tf.float32, shape=(1))
    model = BuildTModel(inputs_placeholder, labels_placeholder, predicted_placeholder, repeated_placeholder)
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
        pred_char = get_predicted_character(data, output_logits[i])
        new_sample[i][1]=pred_char
        inp[i] = data.char_tensor(pred_char)

    for i in range(2, predict_length):
        for j in range(batch_size):
            full_inputs[j] = add_axis_0(inp[j])

        output_logits, next_state = sess.run([model.output_logits_decoded, model.state_final],
        feed_dict={
            inputs_placeholder: full_inputs,
            initial_state_placeholder: next_state
        })
        for j in range(batch_size):
            pred_char = get_predicted_character(data, output_logits[j], 1)
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


def push_to_repo():
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "commit message"])
    subprocess.run(["git", "push"])

print("FINISHED INITIALIZATION\n")


starting_model_number = 0
num_train_steps = 100
num_save = 500
save = 1
load = 1
train = 0
num_evaluate = 5000 # Call evaluate after every num_evaluate iterations
num_strings = 200 # number of strings to generate every num_evaluate iterations

if(load):
    if(rnn_option == GeneratorEnum.simple):
        saver.restore(sess, f"models/simple/train_step_45000_simple/model.ckpt")
    elif(rnn_option == GeneratorEnum.lstm):
        saver.restore(sess, f"models/lstm/train_step_45000_lstm/model.ckpt")
    elif(rnn_option == GeneratorEnum.gru):
        saver.restore(sess, f"models/gru/train_step_45000_gru/model.ckpt")


    
def get_strings():
    list1 = []
    for num in range(2):
        if train == 0:
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
    print("Size of valid list: ", len(valid_list))
    num1 = int(batch_size/2)
    num2 = num1 if batch_size%2 == 0 else num1+1
    try:
        return random.sample(valid_list, num1) + random.sample(valid_list, num2)
#            return random.sample(valid_list, batch_size)
    except:
        print("Too few valid strings")
        return valid_list

def get_raw_strings():
    list1 = []
    if train == 0:
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
    return list1


full_inputs = np.empty(shape = (batch_size, max_string_length))
decoded_output = np.empty(shape = (batch_size, max_string_length))
predicted_array = np.empty(shape = (batch_size))
repeated_value = np.empty(shape = (1))
loss_list = []
predicted_loss_list = []
count_of_strings = {}
num_iterations = 100
for i in range(num_iterations):
    b = get_strings()
    repeated = 0
    for j in b:
        if j not in count_of_strings:
            count_of_strings[j] = 1
        else:
            repeated+=count_of_strings[j]
            count_of_strings[j]+=1
    
    predicted_val1 = fun123(b, len(b))
    for j in range(batch_size):
        inp, target = gen_data.char_tensor('<' + b[j]), gen_data.char_tensor(b[j] + '>') 
        inp, target = pad_input_and_target(inp, target, max_string_length)
        full_inputs[j] = add_axis_0(inp)
        decoded_output[j] = add_axis_0(target)
        predicted_array[j] = predicted_val1[j]
        repeated_value[0] = repeated

    decoded_output = decoded_output.astype(int)
    loss, predicted_loss, total_loss, normal_loss, _= sess.run([model.loss, model.predicted_loss, model.total_loss, model.normal_loss, model.train_op],
    feed_dict={
        inputs_placeholder: full_inputs,
        labels_placeholder: decoded_output,
        predicted_placeholder: predicted_array,
        repeated_placeholder: repeated_value
    })
    completed_iterations = i
    print("Iteration Number ", i)
    print("Loss ", loss)
    print("Predicted loss ", predicted_loss)
    print("Normal loss ", normal_loss)
    print("Predicted loss median", np.median(predicted_array))
    print("Total loss ",total_loss)
    print("Repeated ", repeated_value)
    print("Loss to be exponent: ", np.sum(predicted_array))
    print("Loss to be exponent/ batch size: ", np.sum(predicted_array)/batch_size)
    loss_list.append(loss)
    predicted_loss_list.append(np.average(predicted_array))
    if(len(loss_list) > 5):
        flag = 1
        for index in range(len(loss_list) - 1, len(loss_list) - 6, -1):
            if(predicted_loss_list[index] > early_stop_value):
                flag = 0 
                
        if(flag == 1):
            break


individial_loss, loss = sess.run([model.individual_loss, model.loss],
    feed_dict={
        inputs_placeholder: full_inputs,
        labels_placeholder: decoded_output,
        predicted_placeholder: predicted_array
    })
print("Individual Loss: ", individial_loss)
print("Loss/Seq_len: ", loss/seq_len)


final_list = []
print("Generating Strings")
printProgressBar(0, 400, prefix = 'Progress:', suffix = 'Complete', length = 50)

for i in range(400):
    final_list+=get_raw_strings()
    printProgressBar(i+1, 400, prefix = 'Progress:', suffix = 'Complete', length = 50)

print("Completed Iterations: ", completed_iterations)
print("Num unique strings: ", len(set(final_list)))


print("Writing strings to file")
op_file  = open(f"{output_run_path}/{rnn_option_str}_{bias_option_str}_{property_option_str}-{run_number}.txt", "w")
for i in final_list:
    write_now(op_file, f'{i}\n')
op_file.close()
print("Write to file complete")

print("Saving model")
saver.save(sess, f'gen_models/{rnn_option_str}_{bias_option_str}_{property_option_str}-{run_number}/model.ckpt')