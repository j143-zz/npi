#coding: utf-8

from collections import OrderedDict

import numpy as np
import mxnet as mx

from core import Arguments, Program
from env.addition import Addition
from env.env_config import LEFT, RIGHT, WRITE, MOVE, SWAP

### Program library for Addition


# Can we just return tuple of (r, prog, args)?

def add(env, args):
    (input_1, input_2, carry, output) = env.observation()
    for ptr in env.scratch_pad.pointers:
        print ptr.position
    if input_1 == ' ' and input_2 == ' ' and carry == ' ':
        return None
    call = []
    call.append((ADD1, Arguments()))
    call.append((LSHIFT, Arguments()))
    return call

def add1(env, args):
    (input_1, input_2, carry, output) = env.observation()
    if input_1 == ' ':
        input_1 = 0
    if input_2 == ' ':
        input_2 = 0
    if carry == ' ':
        carry = 0

    res = input_1 + input_2 + carry
    call = []
    if res > 9:
        call.append((ACT, Arguments((4, res % 10, WRITE))))
        call.append((1, CARRY, Arguments()))
    else:
        call.append((1, ACT, Arguments((4, res % 10, WRITE))))
    return call

def carry(env, args):
    call = []
    call.append((ACT, Arguments((3, LEFT, MOVE))))
    call.append((ACT, Arguments((3, 1, WRITE))))
    call.append((1, ACT, Arguments((3, RIGHT, MOVE))))
    return call

def lshift(env, args):
    call = []
    for i in xrange(1, 4):
        call.append((ACT, Arguments((i, LEFT, MOVE))))
    call.append((1, ACT, Arguments((4, LEFT, MOVE))))
    return call

def rshift(env, args):
    call = []
    for i in xrange(1, 4):
        call.append((ACT, Arguments((i, RIGHT, MOVE))))
    call.append((1, ACT, Arguments((4, RIGHT, MOVE))))
    return call

def write(env, args):
    (a_1, a_2, a_3) = args.decode()
    env.scratch_pad.update(a_1-1, a_2)
    return None

def move(env, args):
    (a_1, a_2, a_3) = args.decode()
    print a_1
    if a_2 == LEFT:
        env.scratch_pad.pointers[a_1-1].to_left()
    else:
        env.scratch_pad.pointers[a_1-1].to_right()
    return None

ACT = Program('ACT', None, {'WRITE': write, 'MOVE': move})
LSHIFT = Program('LSHIFT', None, lshift)
RSHIFT = Program('RSHIFT', None, rshift)
CARRY = Program('CARRY', None, carry)
ADD1 = Program('ADD1', None, add1)
ADD = Program('ADD', None, add)












    # get pointers' value, do an addition and then 
