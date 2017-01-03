#coding: utf-8

import numpy as np
from collections import OrderedDict

from config import MAX_ARGS_NUM, ARG_DEPTH, DEFAULT_ARGS
from env.env_config import DEFAULT, LEFT, RIGHT, WRITE, MOVE, SWAP

class Arguments(object):

    """
    In the paper args are represented as an 3-items tuple (a_1, a_2, a_3), we find that each item has its own meaning, namely, a_1 represents the position the program is going to modify. a_2 represents the result, in the addition environment, a_2 is the digit written to the scratch pad if a_3 == write and left or right if a_3 == move(ommitted in paper). a_3 means the action that ACT() should take at current step.

    We encode each arg as an integer between 0~9, by one-hot encoding, the args tuple can be represented with an (MAX_ARGS_NUM, ARG_DEPTH) array. Different value of a_3 will be decoded to more comprehensible form by get_content().
    """
    arg_pos = None
    arg_res = None
    arg_act = None

    def __init__(self, value=DEFAULT_ARGS):
        if value:
            one_hot_args = self.one_hot(value, (MAX_ARGS_NUM, ARG_DEPTH))
            self.args = one_hot_args
            self.arg_pos = self.args[0]
            self.arg_res = self.args[1]
            self.arg_act = self.args[2]
        else:
            self.args = None

    def copy(self):
        new_args = Arguments()
        new_args.args = np.copy(self.args) if isinstance(self.args, np.ndarray) else None
        (new_args.arg_pos, new_args.arg_res, new_args.arg_act) = new_args.args if isinstance(new_args.args, np.ndarray) else (None, None, None)
        return new_args

    def decode(self):
        if isinstance(self.args, np.ndarray):
            a_1 = np.argmax(self.arg_pos)
            a_2 = np.argmax(self.arg_res)
            a_3 = np.argmax(self.arg_act)
            return (a_1, a_2, a_3)
        else:
            return None

    def get_content(self):
        if self.decode():
            (a_1, a_2, a_3) = self.decode()
            if a_3 == WRITE:
                return (a_1, a_2, 'WRITE')
            elif a_3 == MOVE and a_1 != DEFAULT:
                if a_2 == LEFT:
                    return (a_1, 'LEFT')
                elif a_2 == RIGHT:
                    return (a_1, 'RIGHT')
                else:
                    raise ValueError('Unexpected Move Directions!')
            elif a_3 == MOVE and a_1 == DEFAULT:
                if a_2 == LEFT:
                    return ('LEFT')
                elif a_2 == RIGHT:
                    return ('RIGHT')
                else:
                    raise ValueError('Unexpected Move Directions!')
            elif a_3 == SWAP:
                return ('SWAP')
            else:
                raise ValueError('Unexpected Act Type!')
        else:
            return None

    def one_hot(self, value, size, dtype=np.float):
        enc = np.zeros(size, dtype=dtype)
        for i, v in enumerate(value):
            enc[i, v] = 1
        return enc

    def __str__(self):
       return str(self.get_content())

    def __repr__(self):
         return "<Arguments: %s>" % str(self.get_content())


class Program(object):

    def __init__(self, name, args=None, method=None
                 #, sub_progs=None
                ):
        self.name = name
        self.args = args
        self.prog_id = None
        self.method = method
        #self.sub_progs = sub_progs

    def get_content(self, args):
        return "%s" %self.name
        return "%s(%s)" %(self.name, args.get_content())


    def one_hot(self, size, dtype=np.float):
        enc = np.zeros((size,), dtype=dtype)
        enc[self.prog_id] = 1
        return enc

    def __str__(self):
        return self.get_content(self.args)

    def __repr__(self):
        return "<Program: name=%s>" % self.name


