# coding: utf-8
# pylint: disable=too-many-arguments, too-many-locals
"""Definition of various recurrent neural network cells."""
from __future__ import print_function

import bisect
import random
import numpy as np

from mxnet.io import DataIter, DataBatch, DataDesc
from mxnet import ndarray

from config import MAX_ARGS_NUM, ARG_DEPTH
from env.env_config import NUM_OF_ROWS, NUM_OF_COLUMNS, ONE_HOT_DIM

class NPIBucketIter(DataIter):
    """Bucketing iterator for multiple input and output squence model.
    Parameters
    ----------
    data : list of three components of input, [env, prog, args], with each composed of N list, where N is the size of training set.
    label : list of three components of output labels, [end, prog, args], with each composed of N list, where N is the size of training set.
    batch_size : int
        batch_size of data
    invalid_label : int, default -1
        key for invalid label, e.g. <end-of-sentence>
        for array-like input data, invalid_label will be an array with the same size as input, and filled with invalid_label.
    dtype : str, default 'float32'
        data type
    buckets : list of int
        size of data buckets. Automatically generated if None.
    data_name : list of str, default ['curr_env', 'curr_prog', 'curr_args' ]
        name of data
    label_name : list of str, default ['next_end', 'next_prog', 'next_args']
        name of label
    layout : str
        format of data and label. 'NT' means (batch_size, length)
        and 'TN' means (length, batch_size).
    """
    def __init__(self, data, label, batch_size, buckets=None, invalid_label=-1, data_name=['curr_env', 'curr_prog', 'curr_args' ], label_name=['next_end', 'next_prog', 'next_args'], dtype='float32', layout='NTC'):
        super(NPIBucketIter, self).__init__()
        if not buckets:
            buckets = [i for i, j in enumerate(np.bincount([len(s) for s in data[0]]))
                       if j >= batch_size]
        buckets.sort()

        ndiscard = 0
        self.data = [[] for _ in buckets]
        self.label = [[] for _ in buckets]
        for i, sent in enumerate(data[0]):
            buck = bisect.bisect_left(buckets, len(sent))
            if buck == len(buckets):
                ndiscard += 1
                continue
            env_buff = np.full((buckets[buck], NUM_OF_ROWS, NUM_OF_COLUMNS, ONE_HOT_DIM), invalid_label, dtype=dtype)
            env_buff[:len(sent)] = sent
            prog_buff = np.full((buckets[buck],), invalid_label, dtype=dtype)
            prog_buff[:len(sent)] = data[1][i]
            args_buff = np.full((buckets[buck], MAX_ARGS_NUM, ARG_DEPTH), invalid_label, dtype=dtype)
            args_buff[:len(sent)] = data[2][i]
            buff = [env_buff, prog_buff, args_buff]
            self.data[buck].append(buff)

            end_buff = np.full((buckets[buck],), invalid_label, dtype=dtype)
            end_buff[:len(sent)] = label[0][i]
            next_prog_buff = np.full((buckets[buck],), invalid_label, dtype=dtype)
            next_prog_buff[:len(sent)] = label[1][i]
            next_args_buff = np.full((buckets[buck], MAX_ARGS_NUM, ARG_DEPTH), invalid_label, dtype=dtype)
            args_buff[:len(sent)] = label[2][i]
            label_buff = [end_buff, next_prog_buff, next_args_buff]
            self.label[buck].append(label_buff)

        print("WARNING: discarded %d sequences longer than the largest bucket."%ndiscard)

        self.batch_size = batch_size
        self.buckets = buckets
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = dtype
        self.invalid_label = invalid_label
        self.nddata = []
        self.ndlabel = []
        self.major_axis = layout.find('N')
        self.default_bucket_key = max(buckets)

        self.provide_data = [
            ('curr_env', (batch_size, self.default_bucket_key, NUM_OF_ROWS, NUM_OF_COLUMNS, ONE_HOT_DIM)),
            ('curr_prog', (batch_size, self.default_bucket_key)),
            ('curr_args', (batch_size, self.default_bucket_key, MAX_ARGS_NUM, ARG_DEPTH))
        ]

        self.provide_label = [
            ('next_end', (batch_size, self.default_bucket_key)),
            ('next_prog', (batch_size, self.default_bucket_key)),
            ('next_args', (batch_size, self.default_bucket_key, MAX_ARGS_NUM, ARG_DEPTH))
        ]

        # if self.major_axis == 0:
        #     self.provide_data = [(data_name, (batch_size, self.default_bucket_key))]
        #     self.provide_label = [(label_name, (batch_size, self.default_bucket_key))]
        # elif self.major_axis == 1:
        #     self.provide_data = [(data_name, (self.default_bucket_key, batch_size))]
        #     self.provide_label = [(label_name, (self.default_bucket_key, batch_size))]
        # else:
        #     raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")

        self.idx = []
        for i, buck in enumerate(self.data):
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0

        self.reset()

    def reset(self):
        self.curr_idx = 0
        random.shuffle(self.idx)
        for buck in self.data:
            np.random.shuffle(buck)

        self.nddata = []
        self.ndlabel = []
        for buck in self.data:
            if buck:
                ndenv = []
                ndprog = []
                ndargs = []
                for seq in buck:
                    ndenv.append(np.asarray(seq[0], dtype=self.dtype))
                    ndprog.append(np.asarray(seq[1], dtype=self.dtype))
                    ndargs.append(np.asarray(seq[2], dtype=self.dtype))
                self.nddata.append([ndarray.array(np.asarray(ndenv)), ndarray.array(np.asarray(ndprog)), ndarray.array(np.asarray(ndargs))])
            else:
                self.nddata.append(ndarray.array(buck, dtype=self.dtype))
        for buck in self.label:
            if buck:
                ndend = []
                ndprog = []
                ndargs = []
                for seq in buck:
                    ndend.append(np.asarray(seq[0], dtype=self.dtype))
                    ndprog.append(np.asarray(seq[1], dtype=self.dtype))
                    ndargs.append(np.asarray(seq[2], dtype=self.dtype))
                self.ndlabel.append([ndarray.array(np.asarray(ndend)), ndarray.array(np.asarray(ndprog)), ndarray.array(np.asarray(ndargs))])
            else:
                self.ndlabel.append(ndarray.array(buck, dtype=self.dtype))
    def next(self):

        if self.curr_idx == len(self.idx):
            raise StopIteration
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1

        # if self.major_axis == 1:
        #     data = self.nddata[i][j:j+self.batch_size].T
        #     label = self.ndlabel[i][j:j+self.batch_size].T
        #else:
        data = [self.nddata[i][0][j:j+self.batch_size], self.nddata[i][1][j:j+self.batch_size], self.nddata[i][2][j:j+self.batch_size]]
        label = [self.ndlabel[i][0][j:j+self.batch_size], self.ndlabel[i][1][j:j+self.batch_size], self.ndlabel[i][2][j:j+self.batch_size]]
        return DataBatch(data, label, pad=0,
                         bucket_key=self.buckets[i],
                         provide_data=[
                             (self.data_name[0], data[0].shape),
                             (self.data_name[1], data[1].shape),
                             (self.data_name[2], data[2].shape)
                         ],
                         provide_label=[
                             (self.label_name[0], label[0].shape),
                             (self.label_name[1], label[1].shape),
                             (self.label_name[2], label[2].shape)
                         ])



class MultiSequenceBucketIter(DataIter):
    """Bucketing iterator for multiple input and output squence model.
    Parameters
    ----------
    data : list of list of list fo input data
        encoded data
    label : list of list of list of labels
        encoded labels
    batch_size : int
        batch_size of data
    invalid_label : int, default -1
        key for invalid label, e.g. <end-of-sentence>
    dtype : str, default 'float32'
        data type
    buckets : list of int
        size of data buckets. Automatically generated if None.
    data_name : str, default 'data'
        name of data
    label_name : str, default 'softmax_label'
        name of label
    layout : str
        format of data and label. 'NT' means (batch_size, length)
        and 'TN' means (length, batch_size).
    """
    def __init__(self, data, label, batch_size, buckets=None, invalid_label=-1,
                 data_name='data', label_name='softmax_label', dtype='float32',
                 layout='NTC'):
        super(MultiSequenceBucketIter, self).__init__()
        if not buckets:
            buckets = [i for i, j in enumerate(np.bincount([len(s) for s in data]))
                       if j >= batch_size]
        buckets.sort()

        ndiscard = 0
        self.data = [[] for _ in buckets]
        for seg in data:
            for i, sent in enumerate(seg):
                buck = bisect.bisect_left(buckets, len(sent))
                if buck == len(buckets):
                    ndiscard += 1
                    continue
                if seg._dtype == 'int':
                    buff = np.full((buckets[buck],), invalid_label, dtype=dtype)
                    buff[:len(sent)] = sent
                    self.data[buck].append(buff)

        self.data = [np.asarray(i, dtype=dtype) for i in self.data]

        print("WARNING: discarded %d sentences longer than the largest bucket."%ndiscard)

        self.batch_size = batch_size
        self.buckets = buckets
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = dtype
        self.invalid_label = invalid_label
        self.nddata = []
        self.ndlabel = []
        self.major_axis = layout.find('N')
        self.default_bucket_key = max(buckets)

        if self.major_axis == 0:
            self.provide_data = [(data_name, (batch_size, self.default_bucket_key))]
            self.provide_label = [(label_name, (batch_size, self.default_bucket_key))]
        elif self.major_axis == 1:
            self.provide_data = [(data_name, (self.default_bucket_key, batch_size))]
            self.provide_label = [(label_name, (self.default_bucket_key, batch_size))]
        else:
            raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")

        self.idx = []
        for i, buck in enumerate(self.data):
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0

        self.reset()

    def reset(self):
        self.curr_idx = 0
        random.shuffle(self.idx)
        for buck in self.data:
            np.random.shuffle(buck)

        self.nddata = []
        self.ndlabel = []
        for buck in self.data:
            label = np.empty_like(buck)
            label[:, :-1] = buck[:, 1:]
            label[:, -1] = self.invalid_label
            self.nddata.append(ndarray.array(buck, dtype=self.dtype))
            self.ndlabel.append(ndarray.array(label, dtype=self.dtype))

    def next(self):
        if self.curr_idx == len(self.idx):
            raise StopIteration
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1

        if self.major_axis == 1:
            data = self.nddata[i][j:j+self.batch_size].T
            label = self.ndlabel[i][j:j+self.batch_size].T
        else:
            data = self.nddata[i][j:j+self.batch_size]
            label = self.ndlabel[i][j:j+self.batch_size]

        return DataBatch([data], [label],
                         bucket_key=self.buckets[i],
                         provide_data=[(self.data_name, data.shape)],
                         provide_label=[(self.label_name, label.shape)]) 
