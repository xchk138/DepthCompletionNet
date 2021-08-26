import os
from numpy.core.records import fromarrays

from torch._C import dtype
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

curr_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
root_dir = os.path.dirname(curr_dir)
import sys
sys.path.append(root_dir)

import numpy as np
import torch
import torch.nn as nn
from model.resnet import resnet18, resnet34, resnet50, resnet101
from dataset.cifar10 import LoadCIFAR10_Test


def test_ResNet_on_CIFAR10():
    model_name = 'resnet18_cifar10'
    # training standard resnet on dataset CIFAR-10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #net = resnet18(10)
    last_epoc = 50
    net = torch.load('pretrained/resnet18_cifar10-%03d.pth' % last_epoc, map_location=device)

    net.eval()
    net = net.to(device)
    
    data = LoadCIFAR10_Test()
    ncorr = 0
    nall = 0
    for _, x, gt in data:
        x_ = torch.from_numpy(x).to(device)
        pred = net(x_)
        if np.argmax(pred.detach().cpu().numpy()[0]) == gt:
            ncorr += 1
        nall += 1
    print('Final accuracy: {}'.format(ncorr/nall))


if __name__ == '__main__':
    test_ResNet_on_CIFAR10()