# main.py
# the entrance of project : rBN
# rBN: restricted BatchNorm

import model
import dataset
import utils
import profiler


if __name__ == '__main__':
    data_mnist = dataset.MNIST()
    # for plain model
    bn_mnist = model.VGG_C4F2_BN()
    utils.train(bn_mnist, data_mnist, 'mnist_bn.pth')
    utils.eval(bn_mnist, data_mnist, 'mnist_bn.pth')
    profiler.profile(bn_mnist, data_mnist, 'mnist_bn.pth', 'BatchNorm')

    # for model with rBN
    rbn_mnist = model.VGG_C4F2_rBN()
    utils.train(rbn_mnist, data_mnist, 'mnist_rbn.pth')
    utils.eval(rbn_mnist, data_mnist, 'mnist_rbn.pth')
    profiler.profile(rbn_mnist, data_mnist, 'mnist_rbn.pth', 'BatchNorm')
