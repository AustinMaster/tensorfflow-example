#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# File Name: fashion_mnist.py
# Author: Liu Feng
# Created Time : 2019年07月20日 星期六 22时41分08秒
# Description:
"""
import os
import sys
import gzip
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Loads the Fashion-MNIST dataset.
    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).
    """
    base = '/home/liufeng/Machine_Learning/tf/data'
    dirname = os.path.join(base, 'fashion-mnist')
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(dirname + '/' + fname)

    with gzip.open(paths[0], 'rb') as lbpath:
      y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
      x_train = np.frombuffer(
          imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
      y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
      x_test = np.frombuffer(
          imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    print(len(x_train))
    print(x_train.shape)
    print(y_train)
    plt.figure()
    plt.imshow(x_train[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
