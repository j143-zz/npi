# -*- coding: utf-8 -*-

import numpy as np
import mxnet as mx

from environment import Environment, Pointer
from env_config import NUM_OF_ROWS, NUM_OF_COLUMNS, ONE_HOT_DIM


class Addition(Environment):
    """
    Class for Addition Environment
    """

    def __init__(self):
        """
        """
        shape = (NUM_OF_ROWS, NUM_OF_COLUMNS, ONE_HOT_DIM)
        ptrs = [Pointer(pos=(0, 0)), Pointer(pos=(1, 0)), Pointer(pos=(2, 0)), Pointer(pos=(3, 0))]

        super(Addition, self).__init__(shape=shape, content=None, ptrs=ptrs)


    def _reset(self):
        """
        
        """
        self._pad._reset()
        self._ptrs = [Pointer(pos=(0, 0)), Pointer(pos=(1, 0)), Pointer(pos=(2, 0)), Pointer(pos=(3, 0))]


    def __call__(self, input_1, input_2):
        """
        Keyword Arguments:
        input_1 -- 
        input_2 -- 
        """
        self._reset()
        self._input_1 = input_1
        self._input_2 = input_2

        content_asnumpy = np.zeros(self._pad._shape)
        for i in range(len(str(self._input_1))):
            content_asnumpy[0, i, int(str(self._input_1)[::-1][i])] = 1
        for i in range(len(str(self._input_2))):
            content_asnumpy[1, i, int(str(self._input_2)[::-1][i])] = 1

        self._pad._load(mx.nd.array(content_asnumpy))

    @property
    def _one_hot(self):
        """
        """
        return self._pad._content

    @property
    def _ob(self):
        """
        
        """
        res = []
        for ptr in self._ptrs:
            value = self._pad._content.asnumpy()[ptr._pos]
            res.append(np.argmax(value) if not np.array_equal(value, np.zeros((ONE_HOT_DIM, ))) else ' ')
        return res

    @property
    def _curr(self):
        """
        
        """
        curr=[]
        curr_lines = mx.nd.argmax(self._pad._content, axis=2)
        for r in range(NUM_OF_ROWS):
            val = 0
            for c in range(NUM_OF_COLUMNS):
                val += curr_lines[r][c] * 10**c
            curr.append(val.asnumpy()[0])
        return curr

    @property
    def _result(self):
        """
        """
        res = 0
        res_line = mx.nd.argmax(self._pad._content[3], axis=1)
        for c in range(NUM_OF_COLUMNS):
            res += res_line[c] * 10**c
        return res.asnumpy()[0]

    def __str__(self):
        """
        """
        return str(self._pad._content.asnumpy())
