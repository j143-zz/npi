import numpy as np
import mxnet as mx
import random
import time
from npi.env.addition import Addition
from npi.lib import *
from npi.core import *
from npi.generator import *
from npi.npi_io import NPIBucketIter
from npi.config import MAX_ARGS_NUM, ARG_DEPTH, PROG_VEC_SIZE, KEY_VEC_SIZE, MAX_PROG_NUM
from npi.metric import ModifiedPerplexity

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

env_data = []
prog_data = []
args_data = []
end_label = []
prog_label = []
args_label = []

add_env = Addition()

data_gen = Generator()

num_data = 30000
num_train = 20000

batch_size = 100

for i in range(num_data):
    input_1 = random.randint(0, 999999)
    input_2 = random.randint(0, 999999)
    add_env(input_1, input_2)
    data_gen.inference(add_env, ADD, Arguments())
    env_data.append(data_gen.env_data)
    prog_data.append(data_gen.prog_data)
    args_data.append(data_gen.args_data)
    end_label.append(data_gen.end_label)
    prog_label.append(data_gen.prog_label)
    args_label.append(data_gen.args_label)

    data_gen.clear()

data = [env_data[:num_train], prog_data[:num_train], args_data[:num_train]]
label = [end_label[:num_train], prog_label[:num_train], args_label[:num_train]]

val_data = [env_data[num_train:], prog_data[num_train:], args_data[num_train:]]
val_label =  [end_label[num_train:], prog_label[num_train:], args_label[num_train:]] 

buckets = [20, 30, 40, 50, 60, 70]

data_train = NPIBucketIter(data=data, label=label, batch_size=batch_size, buckets=buckets)
data_val =  NPIBucketIter(data=val_data, label=val_label, batch_size=batch_size, buckets=buckets)


f_lstm = mx.rnn.SequentialRNNCell()
for i in range(2):
    f_lstm.add(mx.rnn.LSTMCell(num_hidden=256, prefix='lstm_l%d_'%i))



def sym_gen(bucket_size):
        curr_env = mx.sym.Variable('curr_env')
        curr_prog = mx.sym.Variable('curr_prog')
        curr_args = mx.sym.Variable('curr_args')
        next_end = mx.sym.Variable('next_end')
        next_prog = mx.sym.Variable('next_prog')
        next_args = mx.sym.Variable('next_args')

        prog_memory = mx.sym.arange(start=0, stop=MAX_PROG_NUM, name='prog_memory')

        curr_env_reshaped = mx.sym.Reshape(curr_env, shape=(batch_size * bucket_size, -1))
        curr_args_reshaped = mx.sym.Reshape(curr_args, shape=(batch_size * bucket_size, -1))
        env_args_concat = mx.sym.Concat(curr_env_reshaped, curr_args_reshaped, dim=1, name='s_t')

        #f_enc
        f_enc_fc1 = mx.sym.FullyConnected(data=env_args_concat, num_hidden=256, name='f_enc_fc1')
        f_enc_act1 = mx.sym.Activation(data=f_enc_fc1, act_type='relu', name='f_enc_act1')
        f_enc = mx.sym.FullyConnected(data=f_enc_act1, num_hidden=128, name='f_enc')
        # (batch_size * bucket_size, 128)

        #program embedding
        prog_embedding = mx.sym.Embedding(data=curr_prog, input_dim=MAX_PROG_NUM,
                                           output_dim=PROG_VEC_SIZE, name='prog_embedding')
        key_memory = mx.sym.FullyConnected(data=prog_memory, num_hidden=KEY_VEC_SIZE, name='key_memory')
        prog_embedding_reshaped = mx.sym.Reshape(prog_embedding, shape=(batch_size * bucket_size, -1))
        curr_inputs = mx.sym.Concat(f_enc, prog_embedding_reshaped, dim=1, name='curr_inputs')

        # 2-layer MLP with ReLU activation and linear decoder
        curr_inputs_fc1 = mx.sym.FullyConnected(data=curr_inputs, num_hidden=128, name='curr_inputs_fc1')
        curr_inputs_act1 = mx.sym.Activation(data=curr_inputs_fc1, act_type='relu', name='curr_inputs_act1')
        enc_inputs = mx.sym.FullyConnected(data=curr_inputs_act1, num_hidden=32, name='enc_inputs')
        enc_inputs_reshaped = mx.sym.Reshape(enc_inputs, shape=(batch_size, bucket_size, -1))

        # 2-layer lstm core
        f_lstm.reset()
        outputs, states = f_lstm.unroll(bucket_size, inputs=enc_inputs_reshaped, merge_outputs=True)

        outputs_reshaped = mx.sym.Reshape(outputs, shape=(-1, 256))

        # f_end
        f_end_fc1 = mx.sym.FullyConnected(data=outputs_reshaped, num_hidden=2, name='f_end_fc1')
        next_end_reshaped = mx.sym.Reshape(next_end, shape=(-1,))
        f_end = mx.sym.SoftmaxOutput(data=f_end_fc1, label=next_end_reshaped, name='f_end')

        # f_prog
        f_prog_fc1 = mx.sym.FullyConnected(data=outputs_reshaped, num_hidden=KEY_VEC_SIZE, name='f_prog_fc1')
        f_prog_dot = mx.sym.dot(f_prog_fc1, key_memory, transpose_b=True, name='f_prog_dot')

        #f_prog_choice = mx.sym.argmax(data=f_prog_dot, axis=1, name='f_prog_choice')
        next_prog_reshaped = mx.sym.Reshape(next_prog, shape=(-1,))
        f_prog = mx.sym.SoftmaxOutput(data=f_prog_dot, label=next_prog_reshaped, name='f_prog')

        # f_args
        next_args_reshaped = mx.sym.Reshape(next_args, shape=(-1, MAX_ARGS_NUM, ARG_DEPTH), name='next_args_reshaped')
        next_args_numeric = mx.sym.argmax(data=next_args_reshaped, axis=2, name='next_args_numeric')
        args = mx.sym.SliceChannel(next_args_numeric, axis=1, num_outputs=3)
        f_arg0_fc1 = mx.sym.FullyConnected(data=outputs_reshaped, num_hidden=ARG_DEPTH, name='f_arg0_fc1')
        f_arg1_fc1 = mx.sym.FullyConnected(data=outputs_reshaped, num_hidden=ARG_DEPTH, name='f_arg1_fc1')
        f_arg2_fc1 = mx.sym.FullyConnected(data=outputs_reshaped, num_hidden=ARG_DEPTH, name='f_arg2_fc1')
        arg0_label = mx.sym.Reshape(args[0], shape=(-1,))
        arg1_label = mx.sym.Reshape(args[1], shape=(-1,))
        arg2_label = mx.sym.Reshape(args[2], shape=(-1,))
        f_arg0_softmax = mx.sym.SoftmaxOutput(data=f_arg0_fc1, label=arg0_label, name='f_arg0_softmax')
        f_arg1_softmax = mx.sym.SoftmaxOutput(data=f_arg1_fc1, label=arg2_label, name='f_arg1_softmax')
        f_arg2_softmax = mx.sym.SoftmaxOutput(data=f_arg2_fc1, label=arg2_label, name='f_arg2_softmax')
            # f_arg_argmax = mx.sym.argmax(data=f_arg_softmax, axis=1, name='f_arg%s_argmax' %itr)
        #     f_arg_argmax_int = mx.sym.Cast(data=f_arg_argmax, dtype='int32')
        #     f_arg_one_hot = mx.sym.one_hot(f_arg_argmax_int, depth=10, on_value=1, off_value=0)
        #     f_args_list.append(f_arg_one_hot)
        # f_args_flatten = mx.sym.Concat(f_args_list[0], f_args_list[1], f_args_list[2], dim=1, name='f_args_flatten')
        # f_args = mx.sym.Reshape(f_args_flatten, shape=(batch_size, bucket_size, MAX_ARGS_NUM, ARG_DEPTH))
        f_args_concat = mx.sym.Concat(f_arg0_softmax, f_arg1_softmax, f_arg2_softmax, dim=1, name='f_args_concat')
        f_args = mx.sym.Reshape(f_args_concat, shape=(batch_size, bucket_size, MAX_ARGS_NUM, ARG_DEPTH))
        out = mx.sym.Group([f_end, f_prog, f_args])
        return out, ('curr_env', 'curr_prog', 'curr_args'), ('next_end', 'next_prog', 'next_args')


model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_train.default_bucket_key,
        context             = mx.gpu(0))


# model.bind(data_shapes=data_train.provide_data, label_shapes=data_train.provide_label, for_training=True, force_rebind=True)
# model.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),allow_missing=True, force_init=True)
# model.init_optimizer(kvstore='local', optimizer='adam',optimizer_params={ 'learning_rate': 0.0001, 'wd': 0.000095 })

# toy_env = Addition()
# toy_env(123, 789)

# data_gen.inference(toy_env, ADD, Arguments())
# data = [[data_gen.env_data], [data_gen.prog_data], [data_gen.args_data]]
# label = [[data_gen.end_label],[data_gen.prog_label],[data_gen.args_label]]

# toy_data = NPIBucketIter(data=data, label=label, batch_size=1, buckets=buckets)

# outputs = model.predict(toy_data, num_batch=1)
# argmax_outputs = [mx.nd.argmax(outputs[0], axis=1).asnumpy(), mx.nd.argmax(outputs[1], axis=1).asnumpy(), mx.nd.argmax(outputs[2], axis=3).asnumpy()]
# for i in range(30):
#     print argmax_outputs[0][i], argmax_outputs[1][i], (argmax_outputs[2][0][i][0], argmax_outputs[2][0][i][1], argmax_outputs[2][0][i][2])

# for bucket in buckets:
#     if not bucket in model._buckets:
#         symbol, data_names, label_names = model._sym_gen(bucket)
#         data_shapes = [
#             ('curr_env', (batch_size, bucket, 4, 9, 10)),
#             ('curr_prog', (batch_size, bucket)),
#             ('curr_args', (batch_size, bucket, 3, 10))
#         ]
#         label_shapes = [
#             ('next_end', (batch_size, bucket)),
#             ('next_prog', (batch_size, bucket)),
#             ('next_args', (batch_size, bucket, 3, 10))
#         ]
#         module = mx.module.Module(symbol, data_names, label_names,
#                             logger=model.logger, context=model._context,
#                             work_load_list=model._work_load_list,
#                             fixed_param_names=model._fixed_param_names,
#                             state_names=model._state_names)
#         module.bind(data_shapes, label_shapes, model._curr_module.for_training, model._curr_module.inputs_need_grad, force_rebind=False, shared_module=model._buckets[model._default_bucket_key])
#         model._buckets[bucket] = module


model.fit(
        train_data          = data_train,
        eval_data           = data_val,
        eval_metric         = ModifiedPerplexity(-1),
        kvstore             = 'device',
        optimizer           = 'adam',
        optimizer_params    = { 'learning_rate': 0.0001,
                                'wd': 0.000095 },
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch           = 25,
        batch_end_callback  = mx.callback.Speedometer(100, 25)
        epoch_end_callback  = mx.callback.module_checkpoint(model, 'NPI-model'))

# num_epoch = 1
# eval_metric = ModifiedPerplexity(-1)

# for epoch in range(num_epoch):
#     tic = time.time()
#     eval_metric.reset()
#     while 1:
#         try: next_data_batch = data_train.next()
#         except StopIteration:
#             break
#         model.forward(next_data_batch)
#         model.backward()
#         model.update()
#         #model.update_metric(eval_metric, next_data_batch.label)
#         for name, val in eval_metric.get_name_value():
#             model.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
#         toc = time.time()
#         model.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

