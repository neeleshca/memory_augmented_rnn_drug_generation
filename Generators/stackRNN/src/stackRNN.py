import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import collections
import traceback
from utils import expand

stackControllerState = collections.namedtuple('stackControllerState', ('hidden_state', 'stack'))

class stackRNNCell(tf.contrib.rnn.RNNCell):
    def __init__(self, stack_width, stack_depth, num_units, num_layers):

        # char_size is required to decode the output logits and embedding length is required to construct an embedding
        # using the char tensor

        self.stack_width = stack_width
        self.stack_depth = stack_depth
        self.num_units = num_units
        self.num_layers = num_layers
        #self.char_size = char_size
        self.num_stack_controls = 3
        #self.embedding_length = embedding_length
        #self.embedding_layer = layers.Embedding(self.char_size, self.embedding_length)
        self.stack_controls_layer = layers.Dense(self.num_stack_controls)
        self.stack_input_layer = layers.Dense(self.stack_width)
        #self.decoder = layers.Dense(self.char_size)

        def single_cell(num_units):
            return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias = 1.0)

        self.processor = tf.contrib.rnn.MultiRNNCell([single_cell(self.num_units) for _ in range(self.num_layers)])
        

    def __call__(self, input_val, prev_state):
        prev_hidden_state = prev_state.hidden_state
        prev_stack = prev_state.stack

        prev_LSTM_hidden_state = prev_state[0][-1].h[:]

        #embedded_input = self.embedding_layer(input_val)

        with tf.variable_scope('stack_controller'):
            stack_controls = self.stack_controls_layer(prev_LSTM_hidden_state)
            stack_controls_softmax = tf.nn.softmax(stack_controls)

        with tf.variable_scope('stack_input'):
            stack_input = self.stack_input_layer(tf.expand_dims(prev_LSTM_hidden_state, axis = 1))
            stack_input_tanh = tf.nn.tanh(stack_input)

        stack = self.stack_augmentation(stack_input_tanh, prev_stack, stack_controls_softmax)
        stack_top = tf.expand_dims(stack[:, 0, :], dim=1)
        input_val = tf.expand_dims(input_val, dim=1)
        input_val = tf.squeeze(tf.concat((input_val, stack_top), axis=2), axis=[1])

        output, next_hidden_state = self.processor(input_val, prev_hidden_state)
        #decoded_output = self.decoder(output)

        return output, stackControllerState(
            hidden_state = next_hidden_state,
            stack = stack
        )

        
    def stack_augmentation(self, input_val, prev_stack, controls):
        # This is that creates that new stack
        # Stack is a 3-D tensor of [batch_size, stack_depth, stack_width]

        batch_size = prev_stack.get_shape().as_list()[0]
        
        controls = tf.reshape(controls, [-1, 3, 1, 1])

        zeros_at_the_bottom = tf.zeros([batch_size, 1, self.stack_width])

        # a_push = [1, 2, 3, ... ] from 2-D tensor
        a_push, a_pop, a_no_op = controls[:, 0], controls[:, 1], controls[:, 2]

        stack_down = tf.concat((prev_stack[:, 1:], zeros_at_the_bottom), axis=1)     

        #concatenates the 1st to (stack_height - 1) layers with the tanh'd i/p to the top
        #stack_down has same dimensions as stack (logical e.g.: stack: 1, 2, 3, 4    stack_down = 1, 2, 3, 0.342)
        stack_up = tf.concat((input_val, prev_stack[:, :-1]), axis=1)

        # Computing the new stack using the equation in the paper
        new_stack = a_no_op*prev_stack + a_push*stack_up + a_pop*stack_down

        return new_stack


    def zero_state(self, batch_size, dtype):
        with tf.variable_scope('init', reuse= False):
            hidden_state = self.processor.zero_state(batch_size, dtype)
            stack = expand(
                tf.get_variable('init_stack', [self.stack_depth, self.stack_width],
                initializer=tf.constant_initializer(0)),
                dim = 0, N = batch_size)
        return stackControllerState(
            hidden_state = hidden_state,
            stack = stack)

    @property
    def state_size(self):
        hidden_state = self.processor.state_size
        stack = tf.TensorShape([self.stack_depth * self.stack_width])
        return stackControllerState(
            hidden_state = hidden_state,
            stack = stack
        )

    @property
    def output_size(self):
        return self.processor.output_size

            