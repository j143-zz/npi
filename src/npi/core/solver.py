
import numpy as np

from npi.core.core import Arguments
from env.env_config import LEFT, RIGHT, WRITE, MOVE, SWAP

# # The batch size for training
# batch_size = 32

# # initalize states for LSTM
# init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
# init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
# init_states = init_c + init_h

# # Even though BucketSentenceIter supports various length examples,
# # we simply use the fixed length version here
# data_train = bucket_io.BucketSentenceIter(
#     "./obama.txt", 
#     vocab, 
#     [seq_len], 
#     batch_size,             
#     init_states, 
#     seperate_char='\n',
#     text2id=text2id, 
#     read_content=read_content)

# import mxnet as mx
# import numpy as np
# import logging
# logging.getLogger().setLevel(logging.DEBUG)

# # We will show a quick demo with only 1 epoch. In practice, we can set it to be 100
# num_epoch = 1
# # learning rate 
# learning_rate = 0.01

# # Evaluation metric
# def Perplexity(label, pred):
#     loss = 0.
#     for i in range(pred.shape[0]):
#         loss += -np.log(max(1e-10, pred[i][int(label[i])]))
#     return np.exp(loss / label.size)

# model = mx.model.FeedForward(
#     ctx=mx.gpu(0),
#     symbol=symbol,
#     num_epoch=num_epoch,
#     learning_rate=learning_rate,
#     momentum=0,
#     wd=0.0001,
#     initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

# model.fit(X=data_train,
#           eval_metric=mx.metric.np(Perplexity),
#           batch_end_callback=mx.callback.Speedometer(batch_size, 20),
#           epoch_end_callback=mx.callback.do_checkpoint("obama"))


call = None
call_stack = []
step_list = []
steps = 0
max_steps = 1000
max_depth = 10

def enter_function():
    global call_stack
    global call
    call_stack.append(call or [])
    call = None

def exit_function():
    global call
    global call_stack
    call = call_stack.pop()


def step(env, prog, args):
    global call
    print 'call in step%s' % steps
    print call
    if not call:
        call = prog.method(env, args)
    if call:
        ret = convert_for_step_return(call[0])
        call = call[1:]
    else:
        ret = (1, None, None)
    return ret

def convert_for_step_return(step_values):
        if len(step_values) == 2:
            return (0, step_values[0], step_values[1])
        else:
            return (step_values[0], step_values[1], step_values[2])

def process(env, prog, args):
    global call
    global call_stack

    if not call:
        call = prog.method(env, args)
    if call:
        ret = call[0]
        call = call[1:]
    else:
        ret = (1, None, None)
    return ret

def neural_programming_inference(env, prog, args, depth=0):

    global call
    global call_stack
    global step_list
    global steps
    global max_steps
    global max_depth

    if max_depth < depth or max_steps < steps:
        raise StopIteration()

    enter_function()

    result = (0, None, None)
    while result[0] < 0.5:
        steps += 1
        if max_steps < steps:
            raise StopIteration()

        result = step(env, prog, args.copy())
        print "call:" + str(call)
        if 1:
            step_list.append(((env, prog, args.copy()), result))
            print ((env, prog, args.copy()), result)
        if result[1]: 
            if result[1].name == 'ACT':
                decoded_args = result[2].decode()
                if decoded_args[2] == WRITE:
                    result[1].method['WRITE'](env, result[2].copy())
                elif decoded_args[2] == MOVE:
                    result[1].method['MOVE'](env, result[2].copy())
            else:
              # modify original algorithm
                neural_programming_inference(env, result[1], result[2], depth=depth+1)
    exit_function()
