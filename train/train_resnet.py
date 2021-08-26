import os

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
from dataset.cifar10 import LoadCIFAR10_Train


def train_ResNet_on_CIFAR10():
    model_name = 'resnet18_cifar10'
    # training standard resnet on dataset CIFAR-10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #net = resnet18(10)
    last_epoc = 40
    net = torch.load('resnet18_cifar10-%03d.pth' % last_epoc)

    net.train()
    net = net.to(device)
    #print(net_.dump_info())
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    learning_rate = 5e-5 # 0-20:1e-3, 21-40:1e-4, 40-:5e-5
    #print(net_.parameters)
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    num_epoch = 1000
    batch_size = 8
    save_freq = 1
    log_freq = 100
    batch_counter = 0
    data = LoadCIFAR10_Train(num_epoch, batch_size)
    loss = 0
    for epoc_id, batch_id, batch_x, batch_y in data:
        x_ = torch.from_numpy(batch_x).to(device)
        gt_ = torch.from_numpy(batch_y).to(device)
        pred = net(x_)
        loss_i = loss_fn(pred, gt_)
        opt.zero_grad()
        loss_i.backward()
        opt.step()
        loss += loss_i.item()
        if batch_counter == log_freq - 1:
            print('Epoc#{}/{} Loss: {}'.format(epoc_id + last_epoc, num_epoch + last_epoc, loss/log_freq))
            loss = 0
        batch_counter = (batch_counter + 1) % log_freq
        if epoc_id > 0 and (epoc_id % save_freq == 0) and (batch_id==0):
            torch.save(net, '%s-%03d.pth' % (model_name, epoc_id + last_epoc))
            print('model [%s-%03d.pth] saved.'% (model_name, epoc_id + last_epoc))
    torch.save(net, '%s-%03d.pth' % (model_name, num_epoch + last_epoc))
    print('model [%s-%03d.pth] saved.' % (model_name, num_epoch + last_epoc))
    print('training done.')


if __name__ == '__main__':
    train_ResNet_on_CIFAR10()