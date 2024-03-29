# -*- coding: utf-8 -*-

import numpy as np
import mxnet as mx

from mxnet.metric import EvalMetric
from mxnet import ndarray
class ModifiedPerplexity(EvalMetric):
    """Calculate perplexity
    Parameters
    ----------
    ignore_label : int or None
        index of invalid label to ignore when
        counting. usually should be -1. Include
        all entries if None.
    """
    def __init__(self, ignore_label, axis=-1):
        super(ModifiedPerplexity, self).__init__('ModifiedPerplexity')
        self.ignore_label = ignore_label
        self.axis = axis

    def update(self, labels, preds):
        assert len(labels) == len(preds)
        loss = 0.
        num = 0
        probs = []

        ###handle arg label
        labels[2] = mx.nd.argmax(labels[2], axis=3)
        for label, pred in zip(labels, preds):
            assert label.size == pred.size/pred.shape[-1], \
                "shape mismatch: %s vs. %s"%(label.shape, pred.shape)
            label = label.as_in_context(pred.context).astype(dtype='int32').reshape((label.size,))
            pred = mx.nd.Reshape(pred, shape=(-1, pred.shape[-1]))
            pred = ndarray.pick(pred, label.astype(dtype='int32'), axis=self.axis)
            probs.append(pred)

        for label, prob in zip(labels, probs):
            prob = prob.asnumpy()
            if self.ignore_label is not None:
                ignore = label.asnumpy().flatten() == self.ignore_label
                prob = prob*(1-ignore) + ignore
                num += prob.size - ignore.sum()
            else:
                num += prob.size
            loss += -np.log(np.maximum(1e-10, prob)).sum()

        self.sum_metric += np.exp(loss / num)
        self.num_inst += 1
