#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# File Name: mnist.py
# Author: Liu Feng
# Created Time : 2019年07月20日 星期六 22时41分08秒
# Description:
"""
import numpy as np
def load_data():
    with np.load("/home/liufeng/Machine_Learning/tf/data/mnist.npz") as f:
         x_train, y_train = f['x_train'], f['y_train']
         x_test, y_test = f['x_test'], f['y_test']
    return (x_train,y_train), (x_test,y_test)
