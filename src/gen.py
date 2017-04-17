import npi
import numpy as np
import mxnet as mx
import random

from npi.env.addition import Addition
from npi.lib import *
from npi.core import *
from npi.generator import *
from npi.io import NPIBucketIter


env_data = []
prog_data = []
args_data = []
end_label = []
prog_label = []
args_label = []

add_env = Addition()

data_gen = Generator()

for i in range(150):
    input_1 = random.randint(0, 99999)
    input_2 = random.randint(0, 99999)
    add_env(input_1, input_2)
    data_gen.inference(add_env, ADD, Arguments())
    env_data.append(data_gen.env_data)
    prog_data.append(data_gen.prog_data)
    args_data.append(data_gen.args_data)
    end_label.append(data_gen.end_label)
    prog_label.append(data_gen.prog_label)
    args_label.append(data_gen.args_label)

    data_gen.clear()

data = [env_data[:100], prog_data[:100], args_data[:100]]
label = [end_label[:100], prog_label[:100], args_label[:100]]

val_data = [env_data[100:150], prog_data[100:150], args_data[100:150]]
val_label =  [end_label[100:150], prog_label[100:150], args_label[100:150]] 

buckets = [30, 40, 50, 60, 70, 80]

data_train = NPIBucketIter(data=data, label=label, batch_size=10, buckets=buckets)
data_val =  NPIBucketIter(data=val_data, label=val_label, batch_size=10, buckets=buckets)

# data_train = [mx.nd.array(np.array(env_data)), mx.nd.array(np.array(prog_data)), mx.nd.array(np.array(args_data))]
# label_train = [mx.nd.array(np.array(end_label)), mx.nd.array(np.array(prog_label)), mx.nd.array(np.array(args_label))]

# data_train = mx.io.NDArrayIter(data=data_train, label=label_train, batch_size=1, shuffle=True)

# print data_train.getdata().asnumpy()
