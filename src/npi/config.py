# -*- coding: utf-8 -*-

MAX_ARGS_NUM = 3
ARG_DEPTH = 10 # 0~9, one-hot

PROG_VEC_SIZE = 16
KEY_VEC_SIZE = 8
MAX_PROG_NUM = 10


# a_2 encoding for ACT program. Only needed when a_3 == MOVE
LEFT = 0
RIGHT = 1

# a_3 encoding for ACT program. Because we design to share the ACT program through all the environments, so we put this encoding together and temporarily use integer between 0~9
MOVE = 1
WRITE = 2
