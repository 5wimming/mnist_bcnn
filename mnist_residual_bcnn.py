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

network = tflearn.conv_2d(netData, 64, 3, activation='relu', bias=False)
network = tflearn.residual_bottleneck(network, 3, 16, 64)
network = tflearn.residual_bottleneck(network, 1, 32, 128, downsample=True)
network = tflearn.residual_bottleneck(network, 2, 32, 128)
network = tflearn.residual_bottleneck(network, 1, 64, 256, downsample=True)
network = tflearn.residual_bottleneck(network, 2, 64, 256)

network = tflearn.batch_normalization(network)

network = tflearn.activation(network, 'relu')
print("#########################################3:",network.get_shape())

#network = fully_connected(network, 10, activation='softmax')

#network = regression(network, optimizer='adam', learning_rate=0.01,
#                     loss='categorical_crossentropy', name='target')

# Building Residual Network_B
net = tflearn.conv_2d(netData, 64, 3, activation='relu', bias=False)
net = tflearn.residual_bottleneck(net, 3, 16, 64)
net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 32, 128)
net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 64, 256)

net = tflearn.batch_normalization(net)

net = tflearn.activation(net, 'relu')
print("******************************************3:",net.get_shape())

net = tf.transpose(net, perm=[0,3,1,2])
network = tf.transpose(network, perm=[0,3,1,2])
#print("aaaaaaaaaaaaa11:",net.get_shape())
#print("aaaaaaaaaaaaa22:",network.get_shape())
#net = tflearn.reshape(net,new_shape=[-1,256,49])
#network = tflearn.reshape(network,new_shape=[-1,256,49])

#network = tf.transpose(network, perm=[0,2,1])
#print("aaaaaaaaaaaaa1:",net.get_shape())
#print("aaaaaaaaaaaaa2:",network.get_shape())


bcnn = tf.matmul(net,network)

print("aaaaaaaaaaaaa1:",network.get_shape())
bcnn = tflearn.flatten(bcnn)
print("aaaaaaaaaaaaa2:",bcnn.get_shape())
#net = fully_connected(net, 10, activation='softmax')
#net = tflearn.global_avg_pool(net)


bcnn = fully_connected(bcnn, 10, activation='softmax')
bcnn = tflearn.regression(bcnn, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
						 
# Training
model = tflearn.DNN(bcnn, tensorboard_verbose=0)

model.fit(X,Y, n_epoch=20,
           validation_set=(testX,testY),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')