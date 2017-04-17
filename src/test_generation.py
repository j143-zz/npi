import npi
import numpy as np
import mxnet as mx
import random

from npi.env.addition import Addition
from npi.lib import *
from npi.core import *
from npi.generator import *

add_env = Addition()
gen = Generator()
add_env(189, 266)
gen.inference(add_env, ADD, Arguments())
print gen.seq
gen.clear()


#data_train = mx.io.NDArrayIter(data=data_sent, label=label_sent, batch_size=1, shuffle=True)

#print data_train.getdata().asnumpy()
