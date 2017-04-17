# -*- coding: utf-8 -*-

import numpy as np
import mxnet as mx

class Pointer(object):
    """Class for pointers used in environments."""
    def __init__(self, value=None, pos=None):
        """
        Keyword Arguments:
        value -- (default None)
            value of a pointer
        pos   -- (default None)
            position of a pointer
        """
        self._value = value
        self._pos = pos

    def _to_left(self):
        """Move a pointer one step to left."""
        pos_unpack = list(self._pos)
        pos_unpack[-1] += 1
        self._pos = tuple(pos_unpack)

    def _to_right(self):
        """Move a pointer one step to right."""
        pos_unpack = list(self._pos)
        pos_unpack[-1] -= 1
        self._pos = tuple(pos_unpack)

    def __repr__(self):
        return "<Pointer, value=%s, pos=%s>" %(self._value, self._pos)


class Pad(object):
    """ Class for scratch pad used in environments."""
    def __init__(self, shape=(0, ), content=None):
        """
        Keyword Arguments:
        shape   -- tuple(default (0, ))
            shape of pad
        content -- NDArray(default None)
            content of pad
        """
        self._shape = shape
        if content is None:
            self._content = mx.nd.zeros(shape)
        elif content.shape == shape:
            self._content = content
        else:
            raise ValueError("Inconsistent content shape!")

    def _reset(self):
        """
        """
        self._content = mx.nd.zeros(self._shape)

    def _load(self, content):
        """
        Keyword Arguments:
        content -- NDArray
            the content needed to be loaded by the pad
        """
        if content.shape == self._shape:
            self._content = content
        else:
            raise ValueError("Inconsistent content shape!")


    def _update(self, ptr):
        """
        Update value of pad according to pointer's value.
        Keyword Arguments:
        ptr -- Pointer
        """
        content_asnumpy = self._content.asnumpy()
        content_asnumpy[ptr._pos] = ptr._value
        self._content = mx.nd.array(content_asnumpy)

