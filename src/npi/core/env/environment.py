# coding: utf-8

from random import random
import numpy as np

from npi.interfaces.scratch_pad import Pad

__author__='Cloudyrie'


class Environment(object):
    """
    Environment 
    """
    def __init__(self, shape, content):
        self.scratch_pad = Pad(shape, content)

