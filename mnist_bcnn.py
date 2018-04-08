# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tflearn.data_utils as du
# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
import tensorflow as tf
tf.reset_default_graph()
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

X, mean = du.featurewise_zero_center(X)
testX = du.featurewise_zero_center(testX, mean)

# Building convolutional network_A
netData = input_data(shape=[None, 28, 28, 1], name='input')

network = conv_2d(netData, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)

#network = fully_connected(network, 10, activation='softmax')

#network = regression(network, optimizer='adam', learning_rate=0.01,
#                     loss='categorical_crossentropy', name='target')

# Building Residual Network_B

net = tflearn.conv_2d(netData, 64, 3, activation='relu', bias=False)
# Residual blocks
net = tflearn.residual_bottleneck(net, 3, 16, 64)
net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 32, 128)
net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 64, 256)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)


#net = fully_connected(net, 10, activation='softmax')

axb = tflearn.merge([network, net], mode='elemwise_mul')
axb = fully_connected(axb, 10, activation='softmax')
axb = tflearn.regression(axb, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
						 
# Training
model = tflearn.DNN(axb, tensorboard_verbose=0)

model.fit(X,Y, n_epoch=20,
           validation_set=(testX,testY),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')