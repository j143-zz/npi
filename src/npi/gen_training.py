import numpy as np

from core import Arguments
from config import LEFT, RIGHT, WRITE, MOVE

# class Generator(object):
#     """
#     """
#     def __init__(self):
#         """
#         """
#         self.call = None
#         self.call_stack = []
#         self.data = []
#         self.label = []
#         self.steps = 0
#         self.max_steps = 1000
#         self.max_depth = 10

#     def enter_function(self):
#         self.call_stack.append(self.call or [])
#         self.call = None

#     def exit_function(self):
#         if self.call_stack:
#             self.call = self.call_stack.pop()

#     def step(self, env, prog, args):
#         if not self.call:
#             self.call = prog._method(env, args)
#         if self.call:
#             ret = self.convert_for_step_return(self.call[0])
#             self.call = self.call[1:]
#         else:
#             ret = (1, None, None)
#         return ret

#     def convert_for_step_return(self, step_values):
#         if len(step_values) == 2:
#             return (0, step_values[0], step_values[1])
#         else:
#             return (step_values[0], step_values[1], step_values[2])

#     def neural_programming_inference(self, env, prog, args, depth=0):

#         if self.max_depth < depth or self.max_steps < self.steps:
#             raise StopIteration()

#         self.enter_function()

#         result = (0, None, None)
#         while result[0] < 0.5:
#             self.steps += 1
#             if self.max_steps < self.steps:
#                 raise StopIteration()
#             result = self.step(env, prog, args._copy())
#             if 1:
#                 print env._curr
#                 self.data.append([env._curr, prog._id, args._copy()])
#                 if result[1] is not None and result[2] is not None:
#                     self.label.append([result[0], result[1]._id, result[2]])
#                 else:
#                     self.label.append([result[0], result[1], result[2]]) 
#             if result[1]: 
#                 if result[1]._name == 'ACT':
#                     decoded_args = result[2]._numeric
#                     if decoded_args[2] == WRITE:
#                         result[1]._method['WRITE'](env, result[2]._copy())
#                     elif decoded_args[2] == MOVE:
#                         result[1]._method['MOVE'](env, result[2]._copy())
#                 else:
#                     # modify original algorithm
#                     self.neural_programming_inference(env, result[1], result[2], depth=depth+1)
#             self.exit_function()


#     def clear(self):
#         self.steps = 0
#         self.data = []
#         self.label = []



call = None
call_stack = []
data = []
label = []
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
    if not call:
        call = prog._method(env, args)
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
    global data
    global label
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

        result = step(env, prog, args._copy())
        if 1:
            data.append([env._curr, prog._id, args._copy()])
            if result[1] is not None and result[2] is not None:
                label.append([result[0], result[1]._id, result[2]])
            else:
                label.append([result[0], result[1], result[2]]) 
        if result[1]: 
            if result[1]._name == 'ACT':
                decoded_args = result[2]._numeric
                if decoded_args[2] == WRITE:
                    result[1]._method['WRITE'](env, result[2]._copy())
                elif decoded_args[2] == MOVE:
                    result[1]._method['MOVE'](env, result[2]._copy())
            else:
              # modify original algorithm
                neural_programming_inference(env, result[1], result[2], depth=depth+1)
    exit_function()


def clear():
    global steps
    global data
    global label

    steps = 0
    data = []
    label = []
