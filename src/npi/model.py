# -*- coding: utf-8 -*-

import numpy as np
import mxnet as mx

from config import MAX_ARGS_NUM, ARG_DEPTH, PROG_VEC_SIZE, KEY_VEC_SIZE, MAX_PROG_NUM

class NPIModel(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.f_lstm = mx.rnn.SequentialRNNCell()
        for i in range(2):
            self.f_lstm.add(mx.rnn.LSTMCell(num_hidden=256, prefix='lstm_l%d_'%i))

    def sym_gen(bucket_size):
        curr_env = mx.sym.Variable('curr_env')
        curr_prog = mx.sym.Variable('curr_prog')
        curr_args = mx.sym.Variable('curr_args')
        next_end = mx.sym.Variable('next_end')
        next_prog = mx.sym.Variabel('next_prog')
        next_args = mx.sym.Variabel('next_args')

        prog_memory = mx.sym.arange(start=0, stop=MAX_PROG_NUM, name='prog_memory')

        curr_env_reshaped = mx.sym.Reshape(curr_env, shape=(self.batch_size * bucket_size, -1))
        curr_args_reshaped = mx.sym.Reshape(curr_args, shape=(self.batch_size * bucket_size, -1))
        env_args_concat = mx.sym.Concat(env_reshaped, args_reshaped, dim=1, name='s_t')

        #f_enc
        f_enc_fc1 = mx.sym.FullyConnected(data=env_args_concat, num_hidden=256, name='f_enc_fc1')
        f_enc_act1 = mx.sym.Activation(data=f_enc_fc1, act_type='relu', name='f_enc_act1')
        f_enc = mx.sym.FullyConnected(data=f_enc_act1, num_hidden=128, name='f_enc')
        # (batch_size * bucket_size, 128)

        #program embedding
        prog_embedding = mx.sym.Embedding(data=curr_prog, input_dim=MAX_PROG_NUM,
                                           output_dim=PROG_VEC_SIZE, name='prog_embedding')
        key_memory = mx.sym.FullyConnected(data=prog_memory, num_hidden=KEY_VEC_SIZE, name='key_memory')
        curr_prog_reshaped = mx.sym.Reshape(prog_embedding, shape=(self.batch_size * bucket_size, -1))
        curr_inputs = mx.sym.Concat(f_enc, prog_embedding, dim=1, name='curr_inputs')

        # 2-layer MLP with ReLU activation and linear decoder
        curr_inputs_fc1 = mx.sym.FullyConnected(data=curr_inputs, num_hidden=128, name='curr_inputs_fc1')
        curr_inputs_act1 = mx.sym.Activation(data=curr_inputs_fc1, act_type='relu', name='curr_inputs_act1')
        enc_inputs = mx.sym.FullyConnected(data=curr_inputs_act1, num_hidden=32, name='enc_inputs')
        enc_inputs_reshaped = mx.sym.Reshape(enc_inputs, shape=(self.batch_size, bucket_size, -1))

        # 2-layer lstm core
        self.f_lstm.reset()
        outputs, states = self.f_lstm.unroll(bucket_size, inputs=enc_inputs_reshaped, merge_outputs=True)

        outputs_reshaped = mx.sym.Reshape(outputs, shape=(-1, 256))

        # f_end
        f_end_fc1 = mx.sym.FullyConnected(data=outputs_reshaped, num_hidden=1, name='f_end_fc1')
        f_end_act1 = mx.sym.Activation(data=f_end_fc1, act_type='sigmoid', name='f_end_act1')
        f_end = mx.sym.Reshape(f_end_act1, shape=(self.batch_size, bucket_size))

        # f_prog
        f_prog_fc1 = mx.sym.FullyConnected(data=outputs, num_hidden=KEY_VEC_SIZE, name='f_prog_fc1')
        f_prog_dot = mx.sym.dot(f_prog_fc1, key_memory, transpose_b=True, name='f_prog_dot')
        f_prog_choice = mx.sym.argmax(data=f_prog_dot, axis=1, name='f_prog_choice')
        f_prog = mx.sym.Reshape(f_prog_choice, shape=(self.batch_size, bucket_size))

        # f_args
        next_args_reshaped = mx.sym.Reshape(next_args, shape=(-1, MAX_ARGS_NUM, ARG_DEPTH), name='next_args_reshaped')
        next_args_numeric = mx.sym.argmax(data=next_args_reshaped, axis=2, name='next_args_numeric')
        (arg_0, arg_1, arg_2) = mx.sym.sliceChannel(next_args_numeric, axis=1, num_outputs=3)
        f_args_list = []
        for itr in range(MAX_ARGS_NUM):
            f_arg_fc1 = mx.sym.FullyConnected(data=outputs_reshaped, num_hidden=ARG_DEPTH, name='f_arg_fc1')
            if itr == 0:
                arg_label = mx.sym.Reshape(arg_0, shape=(-1,))
            elif itr == 1:
                arg_label = mx.sym.Reshape(arg_1, shape=(-1,))
            else:
                arg_label = mx.sym.Reshape(arg_2, shape=(-1,))
            f_arg_softmax = mx.sym.SoftmaxOutput(data=f_arg, label=arg_label, name='f_arg_softmax')
            f_arg_argmax = mx.sym.argmax(data=f_arg_softmax, axis=1, name='f_arg%s_argmax' %f_arg)
            f_arg_argmax_int = mx.sym.Cast(data=f_arg_argmax, dtype='int32')
            f_arg_one_hot = mx.sym.one_hot(f_arg_argmax_int, depth=10, on_value=1, off_value=0)
            f_args_list.append(f_arg_one_hot)
        f_args_flatten = mx.sym.Concat(f_args_list[0], f_args_list[1], f_args_list[2], dim=1, name='f_args_flatten')
        f_args = mx.sym.Reshape(f_args_flatten, shape=(self.batch_size, bucket_size, MAX_ARGS_NUM, ARG_DEPTH))

        return [f_end, f_prog, f_args], ('curr_env', 'curr_prog', 'curr_args'), ('next_end', 'next_prog', 'next_args')
