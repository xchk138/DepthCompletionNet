from os import path
import numpy as np
import cv2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def make_onehot(ids, nclass):
    onehot = np.zeros([len(ids), nclass], dtype=np.float32)
    for i in range(len(ids)):
        onehot[i, ids[i]] = 1.0
    return onehot

# Natively, the cifar10 dataset use a NCHW format
def LoadCIFAR10_Train(nepoc, batch_size, onehot=False):
    patch_size = 10000
    n_patch = 5
    h = 32
    w = 32
    c = 3
    nclass = 10
    images = np.zeros([n_patch*patch_size, c, h, w], np.uint8)
    labels = np.zeros([n_patch*patch_size], np.int64)

    for i in range(n_patch):
        data_path = 'E:/Gits/Datasets/CIFAR10/data_batch_{}'.format(i+1)
        patch = unpickle(data_path)
        assert len(patch[b'labels']) == len(patch[b'data'])
        images[patch_size*i: patch_size*i + patch_size] = np.reshape(patch[b'data'], (patch_size, c, h, w))
        labels[patch_size*i: patch_size*i + patch_size] = patch[b'labels']

    '''
    # check if the image is okay
    for i in range(100):
        im = np.transpose(images[i], (1,2,0))
        im = cv2.resize(im, (256, 256))
        cv2.imshow('demo', im)
        cv2.waitKey(1000)
    '''

    # calculate the mean and std 
    x_mean = np.array([0.49139968, 0.48215827, 0.44653124], np.float32)
    x_std = np.array([0.24703233, 0.24348505, 0.26158768], np.float32)
    x_mean = np.reshape(np.stack([x_mean]*(w*h), axis=1), (3,32,32))
    x_std = np.reshape(np.stack([x_std]*(w*h), axis=1), (3,32,32))

    nall = images.shape[0]
    nbatch = nall//batch_size
    seq = np.random.permutation(np.arange(nall))
    # begin a coroutine
    for epoc_id in range(nepoc):
        for batch_id in range(nbatch):
            batch_x = np.float32(images[seq[batch_id*batch_size:(batch_id+1)*batch_size]])
            batch_x = (batch_x / 255.0 - x_mean) / x_std
            if onehot:
                batch_y = make_onehot(labels[seq[batch_id*batch_size:(batch_id+1)*batch_size]], nclass)
            else:
                batch_y = labels[seq[batch_id*batch_size:(batch_id+1)*batch_size]]
            yield (epoc_id, batch_id, batch_x, batch_y)
        seq = np.random.permutation(np.arange(nall))
    

def LoadCIFAR10_Test():
    patch_size = 10000
    h = 32
    w = 32
    c = 3
    nclass = 10
    images = np.zeros([patch_size, c, h, w], np.uint8)
    labels = np.zeros([patch_size], np.int64)

    data_path = 'E:/Gits/Datasets/CIFAR10/test_batch'
    patch = unpickle(data_path)
    assert len(patch[b'labels']) == len(patch[b'data'])
    images[:,:,:,:] = np.reshape(patch[b'data'], (patch_size, c, h, w))
    labels[:] = patch[b'labels']

    '''
    # check if the image is okay
    for i in range(100):
        im = np.transpose(images[i], (1,2,0))
        im = cv2.resize(im, (256, 256))
        cv2.imshow('demo', im)
        cv2.waitKey(1000)
    '''

    # calculate the mean and std 
    x_mean = np.array([0.49139968, 0.48215827, 0.44653124], np.float32)
    x_std = np.array([0.24703233, 0.24348505, 0.26158768], np.float32)
    x_mean = np.reshape(np.stack([x_mean]*(w*h), axis=1), (3,32,32))
    x_std = np.reshape(np.stack([x_std]*(w*h), axis=1), (3,32,32))

    nall = images.shape[0]
    # begin a coroutine
    for i in range(nall):
        x = np.expand_dims(np.float32(images[i]), axis=0)
        x = (x / 255.0 - x_mean) / x_std
        y = labels[i]
        yield (i, x, y)