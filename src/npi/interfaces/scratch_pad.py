#coding: utf-8

import numpy as np

###Currently specified for Addition environment, need to make some changes for generation afterwards.

class Pad(object):
    def __init__(self, shape, content):
        self.shape = shape
        self.content = content
        self.pointers = []
        for i in xrange(0, shape[0]):
            ptr = Pointer([i, 0])
            self.pointers.append(ptr)

    def load(self, content):
        self.content = content

    def update(self, pos, res):
        position = self.pointers[pos].position
        one_hot_res = np.zeros((self.shape[-1], ))
        one_hot_res[res] = 1
        self.content[position] = one_hot_res


class Pointer(object):
    def __init__(self, position):
        self.position = position

    def to_left(self):
        self.position = (self.position[0], self.position[1] + 1)

    def to_right(self):
        self.position = (self.position[0], self.position[1] - 1)

