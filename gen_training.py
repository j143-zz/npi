import random
import npi
import mxnet as mx
from npi.core.env.addition import Addition
from npi.core.lib import *
from npi.core.core import *
from npi.core.solver import *

global steps
global step_list

f = open("./data/training.txt", "w")
f.write("{\n")
add_env = Addition()
for i in range(1000):
    input_1 = random.randint(0, 999)
    input_2 = random.randint(0, 999)
    add_env.execute([input_1, input_2])
    neural_programming_inference(add_env, ADD, Arguments())
    f.write("("+str(input_1)+","+str(input_2)+"):"+str(step_list))
    if i < 999:
        f.write(",")
    clear()
f.write("}\n")
