# -*- coding: utf-8 -*-

import numpy as np
import mxnet as mx

from utils import Pad, Pointer

class Environment(object):
    """
    Abstract base class for Environment
    """

    def __init__(self,
                 shape=(0, ), content=None,
                 ptrs=[]):
        """
        Keyword Arguments:
        shape   -- tuple
            shape of pad for particular environment
        content -- (default None) NDArray
            content of pad
        ptrs -- (default []) list of Pointer
        """
        self._pad = Pad(shape=shape, content=content)
        self._ptrs = ptrs

        self._pad._reset()

    @property
    def _one_hot(self):
        """
        """
        raise NotImplementedError()
