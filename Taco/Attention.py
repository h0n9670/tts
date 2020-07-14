from hparams import hparams as hps
from tensorflow.python.ops import rnn_cell_impl, array_ops, math_ops
from tensorflow.python.framework.tensor_shape import *
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, RNN, Conv1D, Dense, Layer

import os
import sys
rootdir_name = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), '../params')
sys.path.append(rootdir_name)
# print(rootdir_name)
# print(sys.path)


class Decoderlstm:
    def __init__(self):

        self.layers = hps.DecoderRNN_layers
        self.size = hps.DecoderRNN_size

        self.lstm_list = [LSTMCell(self.size) for _ in range(self.layers)]

        self.lstm_cell = RNN(
            self.lstm_list, return_state=True, return_sequences=True)

    def build(self, input_shape):
        self.lstm_cell.build(input_shape)
        self.trainable_weights = self.lstm_cell._trainable_weights
        self.weights = self.lstm_cell.weights

    def compute_output_shape(self, inputs):
        return self.lstm_cell.compute_output_shape(inputs)

    def get_initial_state(self, inputs):
        return self.lstm_cell.get_initial_state(inputs)

    def __call__(self, inputs, initial_state):
        return self.lstm_cell(inputs, initial_state=initial_state)

#######################################################################################################


class LocationSensitiveAttentionLayer(Layer):
    def __init__(self):
        super(LocationSensitiveAttentionLayer, self).__init__()
        self.units = hps.LSA_dim
        self.filters = hps.LSA_filters
        self.kernel = hps.LSA_kernel
        self._cumulate = True

        self.location_convolution = Conv1D(
            filters=self.filters, kernel_size=self.kernel, padding='same', bias_initializer='zeros')
        self.location_layer = Dense(
            self.units, use_bias=False)
        self.query_layer = Dense(self.units, use_bias=False)
        self.memory_layer = Dense(self.units, use_bias=False)

        self.rnn_cell = Decoderlstm()

        self.values = None 

        self.keys = None

    def build(self, input_shape):
        enc_out_seq, dec_out_seq = input_shape
        self.v_a = self.add_weight(name='V_a',
                                   shape=(self.units,),
                                   initializer='uniform',
                                   trainable=True)
        self.b_a = self.add_weight(name='b_a',
                                   shape=(self.units,),
                                   initializer='uniform',
                                   trainable=True)
        if self.memory_layer:
            self.memory_layer.build(enc_out_seq)
            self._trainable_weights += self.memory_layer._trainable_weights
        if self.query_layer:
            if not self.query_layer.built:
                if self.rnn_cell:
                    self.query_layer.build(
                        self.rnn_cell.compute_output_shape(dec_out_seq)[0])
                else:
                    self.query_layer.build(dec_out_seq)
            self._trainable_weights += self.query_layer._trainable_weights
        if self.rnn_cell:
            rnn_input_shape = (
                enc_out_seq[0], 1, dec_out_seq[-1] + enc_out_seq[-1])
            self.rnn_cell.build(rnn_input_shape)
            self._trainable_weights += self.rnn_cell.weights

        conv_input_shape = (enc_out_seq[0], enc_out_seq[1], 1)
        location_input_shape = (enc_out_seq[0], enc_out_seq[1], self.filters)
        self.location_convolution.build(conv_input_shape)
        self.location_layer.build(location_input_shape)

        self._trainable_weights += self.location_convolution._trainable_weights
        self._trainable_weights += self.location_layer._trainable_weights

        super(LocationSensitiveAttentionLayer, self).build(input_shape)

    def call(self, inputs, verbose=False):

        encoder_out_seq, decoder_out_seq = inputs
  
        values = encoder_out_seq
        keys = self.memory_layer(values) if self.memory_layer else values

        def energy_step(query, states):
            previous_alignments = states[0]
            if self.rnn_cell:
                c_i = states[1]
                cell_state = states[2:]

                lstm_input = K.concatenate([query, c_i])
                lstm_input = K.expand_dims(lstm_input, 1)

                lstm_out = self.rnn_cell(lstm_input, initial_state=cell_state)
                lstm_output, new_cell_state = lstm_out[0], lstm_out[1:]
                query = lstm_output

            processed_query = self.query_layer(
                query) if self.query_layer else query

            expanded_alignments = K.expand_dims(previous_alignments, axis=2)

            f = self.location_convolution(expanded_alignments)

            processed_location_features = self.location_layer(f)


            e_i = K.sum(self.v_a * K.tanh(keys + processed_query +
                                          processed_location_features + self.b_a), [2])

            e_i = K.softmax(e_i)

            if self._cumulate:
                next_state = e_i + previous_alignments
            else:
                next_state = e_i


            if self.rnn_cell:
                new_c_i, _ = context_step(e_i, [c_i])

                return e_i, [next_state, new_c_i, *new_cell_state]
            return e_i, [next_state]

        def context_step(inputs, states):

            alignments = inputs
            expanded_alignments = K.expand_dims(alignments, 1)


            c_i = math_ops.matmul(expanded_alignments, values)
            c_i = K.squeeze(c_i, 1)


            return c_i, [c_i]

        def create_initial_state(inputs, hidden_size):
            fake_state = K.zeros_like(inputs)
            fake_state = K.sum(fake_state, axis=[1, 2])
            fake_state = K.expand_dims(fake_state)
            fake_state = K.tile(fake_state, [1, hidden_size])
            return fake_state

        def get_fake_cell_input(fake_state_c):
            fake_input = K.zeros_like(decoder_out_seq)[:, 0, :]
            fake_input = K.concatenate([fake_state_c, fake_input])
            fake_input = K.expand_dims(fake_input, 1)
            return fake_input

        fake_state_c = create_initial_state(values, values.shape[-1])
        fake_state_e = create_initial_state(values, K.shape(values)[1])
        if self.rnn_cell:
            cell_initial_state = self.rnn_cell.get_initial_state(
                get_fake_cell_input(fake_state_c))
            initial_states_e = [fake_state_e,
                                fake_state_c, *cell_initial_state]
        else:
            initial_states_e = [fake_state_e]


        last_out, e_outputs, _ = K.rnn(energy_step,
                                       decoder_out_seq,
                                       initial_states_e)

        c_outputs = math_ops.matmul(e_outputs, values)

        return [c_outputs, e_outputs]

    def comute_output_shape(self, input_shape):
        return [
            (input_shape[1][0], input_shape[1][1], input_shape[1][2]),
            (input_shape[1][0], input_shape[1][1], input_shape[0][1])
        ]
