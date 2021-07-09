import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

class MSConv2d(nn.Module):
    def __int__(self, in_channels, out_channels, ksize, stride, padding, pad_val=0, weights=None):
        assert(in_channels > 0)
        assert(out_channels > 0)
        assert(ksize > 0)
        assert(stride > 0)
        assert(padding=='same' or padding=='valid')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.pad_val = pad_val

        if weights:
            assert(isinstance(weights, np.ndarray)
            assert(weights.shape[0] == out_channels)
            assert(weights.shape[1] == weights.shape[2] == ksize)
            assert(weights.shape[2] == in_channels)
            self.weights = weights
        pass
    def forward(self, inputs):
        pass
    def dump_info(self):
        print(self.ksizes)