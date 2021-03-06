import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# @param input_shape: must be 4-elements list
# as the conv2d in pytorch is in NCHW channel-order
def conv2d_get_output_shape_torch(input_shape, ksize, stride, padding, dilation):
    assert len(input_shape) == 4
    size_in = input_shape[2:4]
    params = [ksize, stride, padding, dilation]
    for i in range(len(params)):
        if isinstance(params[i], list) or isinstance(params[i], tuple):
            assert len(params[i]) == 2
        else:
            assert isinstance(params[i], int)
            params[i] = [params[i], params[i]]
    ksize, stride, padding, dilation = params
    size_out = [(size_in[i] + 2*padding[i] - dilation[i]*(ksize[i]-1) - 1) // stride[i] + 1 for i in [0,1]]
    return (*(input_shape[:2]), *size_out)


class MSConv2d(nn.Module):
    '''
    @param in_channels: input channels
    @param out_channels: output channels
    @param ksize: kernel size
    @param stride: same as plain convolution, step size
    @param scale: (*)pyramid scaling down factor, like 2.
    @param padding: the same as plain convolution
    @param weight: setting the weight from numpy array
    '''
    def __init__(
        self
        ,in_channels
        ,out_channels
        ,kernel_size
        ,stride=1
        ,scale=2
        ,padding=0
        ,bias=True
        ):
        super().__init__()
        ksize = kernel_size
        assert(in_channels > 0)
        assert(out_channels > 0)
        assert(ksize > 0)
        assert(stride > 0)

        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, ksize, stride, padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, ksize, 1, 1, bias=bias)

        self.conv2.weight = self.conv1.weight
        self.conv2.bias = self.conv1.bias
        
    def forward(self, x):
        # do the strided convolution as the main part of feature extraction
        y = self.conv1(x)
        h_in, w_in = x.shape[2:4]
        h_out, w_out = y.shape[2:4]
        # determine depth of the pyramid to build
        assert self.scale > 1
        depth_h = math.floor(math.log(float(h_in) / self.conv1.kernel_size[0]) / math.log(self.scale))
        depth_w = math.floor(math.log(float(w_in) / self.conv1.kernel_size[1]) / math.log(self.scale))
        depth = int(min(depth_h, depth_w))
        if depth > 0:
            # calculate sizes of pyramid to reshape
            sizes_in = [None] * depth
            sizes_in[0] = (w_in//self.scale, h_in//self.scale)
            for i in range(1, depth):
                sizes_in[i] = (sizes_in[i-1][0]//self.scale, sizes_in[i-1][1]//self.scale)

            # generate a pyramid by a given scale to downsample source image
            for i in range(depth):
                # resize input feature maps into desired scale
                x_scaled = F.interpolate(x, size=sizes_in[i], mode='bilinear')
                y_scaled = self.conv2(x_scaled)
                #print(y_scaled.size())
                # resize back to original size
                y_scaled = F.interpolate(y_scaled, size=(w_out, h_out), mode='bilinear')
                y += y_scaled
        
        return y

    def dump_info(self):
        return {
            'scale': self.scale
            , 'conv1': self.conv1
            , 'conv2': self.conv2
            , 'weight': self.conv1.weight
            , 'bias': self.conv1.bias
            }