# -*- coding: utf-8 -*-

import numpy as np
import mxnet as mx

from config import MAX_ARGS_NUM, ARG_DEPTH, LEFT, RIGHT, WRITE, MOVE

class Arguments(object):

    """
    In the paper args are represented as an 3-items tuple (a_1, a_2, a_3), we find that each item has its own meaning, namely, a_1 represents the position the program is going to modify. a_2 represents the result, in the addition environment, a_2 is the digit written to the scratch pad if a_3 == write and left or right if a_3 == move(ommitted in paper). a_3 means the action that ACT() should take at current step.

    We encode each arg as an integer between 0~9, by one-hot encoding, the args tuple can be represented with an (MAX_ARGS_NUM, ARG_DEPTH) array. Different value of a_3 will be decoded to more comprehensible form by args._parse.
    """

    def __init__(self, value=None):
        if value is not None:
            (self._arg_pos, self._arg_res, self._arg_act) = value
        else:
            (self._arg_pos, self._arg_res, self._arg_act) = (None,) * 3

    def _copy(self):
        new_args = Arguments()
        new_args._arg_pos, new_args._arg_res, new_args._arg_act = \
        self._arg_pos, self._arg_res, self._arg_act

        return new_args

    @property
    def _one_hot(self):
        """ """
        one_hot_args = mx.nd.zeros((MAX_ARGS_NUM, ARG_DEPTH))
        if self._arg_pos is not None and self._arg_res is not None and self._arg_act is not None:
            one_hot_args[0, self._arg_pos] = 1
            one_hot_args[1, self._arg_res] = 1
            one_hot_args[2, self._arg_act] = 1
        return one_hot_args

    @property
    def _numeric(self):
        """Return the numerical form of Arguments"""
        return (self._arg_pos, self._arg_res, self._arg_act)

    @property
    def _parse(self):
        if self._arg_pos is not None and self._arg_res is not None and self._arg_act is not None:
            if self._arg_act == WRITE:
                return (self._arg_pos, self._arg_res, 'WRITE')
            elif self._arg_act == MOVE:
                if self._arg_res == LEFT:
                    return (self._arg_pos, 'LEFT')
                elif self._arg_res == RIGHT:
                    return (self._arg_pos, 'RIGHT')
                else:
                    raise ValueError('Unexpected Move Directions!')
            else:
                raise ValueError('Unexpected Act Type!')
        else:
            return None

    def __str__(self):
       return str(self._parse)

    def __repr__(self):
         return "<Arguments: %s>" % str(self._parse)



class Program(object):

    Counter = 0

    def __init__(self, name, args=None, method=None):
        self._name = name
        self._args = args
        self._id = Program.Counter
        self._method = method

        Program.Counter += 1

    @property
    def _parse(self):
        if self._args:
            return "%s%s" %(self._name, self._args._parse)
        return "%s" %self._name

    def __call__(self, env, args):
        """
        Keyword Arguments:
        env  -- Environment
            Particular description of environment the program fits to.
        args -- Arguments
            Arguments that program needs.
        """
        return self._method(env, args)

    def __str__(self):
        return self._parse

    def __repr__(self):
        return "<Program: name=%s, id=%s>" %(self._name, self._id)
