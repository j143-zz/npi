
import numpy as np

from environment import Environment
from env_config import NUM_OF_ROWS, NUM_OF_COLUMNS, ONE_HOT_DIM


###NOTICE: the arguments for position were passed starting from 1! please be carefull when modify the action of pointers!

class Addition(Environment):

    def __init__(self):  #inputs has the form of [input_1, input_2]
        shape = (NUM_OF_ROWS, NUM_OF_COLUMNS, ONE_HOT_DIM)
        init_content = np.zeros(shape)
        init_content[:, :, 10] = 1
        super(Addition, self).__init__(shape, init_content)

        #self.prog_set = self.gen_prog_set()

    def execute(self, inputs):
        self.reset()
        encoded_content = self.encode(inputs)
        self.scratch_pad.load(encoded_content)

    def decode(self):
        rows = np.argmax(self.scratch_pad.content, axis = 2)
        empty = np.ones((NUM_OF_COLUMNS,)) * 10
        dec = []
        for row in rows:
            if not np.array_equal(row, empty):
                row[row==10] = 0
                res = 0
                for c in xrange(NUM_OF_COLUMNS):
                    res += row[c] * 10**c
            else:
                res = ''
            dec.append(res)
        return dec


    def encode(self, inputs):
        augmented_content = inputs + [None, None]
        if len(augmented_content) != NUM_OF_ROWS:
            raise TypeError('content should include 2 inputs.\(%s given\)' % len(inputs))
        elif len(str(augmented_content[0])) > NUM_OF_COLUMNS or len(str(augmented_content[1])) > NUM_OF_COLUMNS :
            raise IndexError('Unexpected inputs: exeed max num of columns.')
        else:
            enc = []
            for item in augmented_content:
                row = []
                str_of_item = (9 - len(str(item))) * ' ' + str(item) if item != None else 9 * ' '
                for ch in str_of_item[::-1]:
                    one_hot = [0] * ONE_HOT_DIM
                    one_hot[int(ch) if ch != ' ' else 10] = 1
                    row.append(one_hot)
                enc.append(row)
            return np.array(enc)

    #Set all columns to white space and shift all pointers to the rightmost
    def reset(self):
        self.scratch_pad.content = np.zeros_like(self.scratch_pad.content)
        self.scratch_pad.content[:, :, 10] = 1
        for ptr in self.scratch_pad.pointers:
            ptr.position = (ptr.position[0], 0)  #0 represents rightmost

    def observation(self):
        ret = []
        for ptr in self.scratch_pad.pointers:
            value = np.argmax(self.scratch_pad.content[ptr.position])
            ret.append(value if value != 10 else ' ')
        return ret

    # def register_prog(self, prog):
    #     prog_id = len(self.prog_set)
    #     prog.prog_id = prog_id
    #     self.prog_set.append(prog)

    # def gen_prog_set(self):
    #     self.register_prog(ADD)
    #     self.register_prog(ADD1)
    #     self.register_prog(CARRY)
    #     self.register_prog(LSHIT)
    #     self.register_prog(RSHIFT)
    #     self.register_prog(ACT)

    def __str__(self):
        decode = self.decode()
        formalized_content = 'Addition: \n' + (10 - len(str(decode[0]))) * ' ' + str(decode[0]) + '\n' + '+' + (9 - len(str(decode[1]))) * ' ' + str(decode[1]) + '\n' + 10 * '-' + '\n' + (10 - len(str(decode[2]))) * ' ' + str(decode[2]) + '\n' + (10 - len(str(decode[3]))) * ' ' + str(decode[3]) + '\n'
        return formalized_content

    def __repr__(self):
        decode = self.decode()
        current_exec = str(decode[0]) + ' + ' +  str(decode[1]) 
        return "<Addition Environment: shape=%s, current_executions=(%s)>" % (self.scratch_pad.shape, current_exec)
