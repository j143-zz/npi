# -*- coding: utf-8 -*-

import numpy as np
import mxnet as mx
from core import Arguments
from config import MAX_ARGS_NUM, ARG_DEPTH, LEFT, RIGHT, WRITE, MOVE

class Generator(object):
    """
    """
    def __init__(self):
        """
        
        """
        self.call = None
        self.call_stack = []
        self.env_data = []
        self.prog_data = []
        self.args_data = []
        self.end_label = []
        self.prog_label = []
        self.args_label = []
        self.seq = []
        self.steps = 0
        self.max_steps = 1000
        self.max_depth = 10

    def enter_function(self):
        """
        
        """
        self.call_stack.append(self.call or [])
        self.call = None

    def exit_function(self):
        """
        
        """
        self.call = self.call_stack.pop()

    def step(self, env, prog, args):
        """
        """
        if not self.call:
            self.call = prog._method(env,args)
        if self.call:
            ret = self.convert_for_step_return(self.call[0])
            self.call = self.call[1:]
        else:
            ret = (-1, -1, np.full((MAX_ARGS_NUM, ARG_DEPTH), -1, dtype='float32'))
        return ret

    def convert_for_step_return(self, step_values):
        """
        Keyword Arguments:
        step_values -- 
        """
        if (len(step_values)) == 2:
            return (0, step_values[0], step_values[1])
        else:
            return (step_values[0], step_values[1], step_values[2])

    def inference(self, env, prog, args, depth=0):
        """
        """
        if self.max_depth < depth or self.max_steps < self.steps:
            raise StopIteration()
        self.enter_function()

        result = (0, -1, np.full((MAX_ARGS_NUM, ARG_DEPTH), -1, dtype='float32'))
        while result[0] < 0.5 and result[0] >= 0:
            self.steps += 1
            if self.max_steps < self.steps:
                raise StopIteration()
            result = self.step(env, prog, args._copy())
            self.env_data.append(env._one_hot.asnumpy())
            self.prog_data.append(prog._id)
            self.args_data.append(args._copy()._one_hot.asnumpy())
            self.end_label.append(result[0])
            if result[0] != -1:
                self.prog_label.append(result[1]._id)
                self.args_label.append(result[2]._copy()._one_hot.asnumpy())
                self.seq.append([(env, prog, args._copy()), (result[0], result[1], result[2]._copy())])
            else:
                self.prog_label.append(result[1])
                self.args_label.append(result[2])
                self.seq.append([(env, prog, args._copy()), (result[0], result[1], result[2])])
            if result[1] != -1:
                if result[1]._name == 'ACT':
                    decoded_args = result[2]._numeric
                    if decoded_args[2] == WRITE:
                        result[1]._method['WRITE'](env, result[2]._copy())
                    elif decoded_args[2] == MOVE:
                        result[1]._method['MOVE'](env, result[2]._copy())
                else:
                    self.inference(env, result[1], result[2]._copy(), depth=depth+1)

        self.exit_function()

    def clear(self):
        """
        """
        self.call = None
        self.call_stack = []
        self.env_data = []
        self.prog_data = []
        self.args_data = []
        self.end_label = []
        self.prog_label = []
        self.args_label = []
        self.seq = []
        self.steps = 0
