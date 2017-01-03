# coding: utf-8


#Environment config for addition environment
NUM_OF_ROWS = 4     # Input1, Input2, Carry, Output
NUM_OF_COLUMNS= 9   # number of columns
ONE_HOT_DIM = 11  # number of characters(0~9 digits) and white space, per cell. one-hot-encoding

DEFAULT  = 10









# a_2 encoding for ACT program. Only needed when a_3 == MOVE
LEFT = 0
RIGHT = 1

# a_3 encoding for ACT program. Because we design to share the ACT program through all the environments, so we put this encoding together and temporarily use integer between 0~9
WRITE = 0
MOVE = 1
SWAP = 2
