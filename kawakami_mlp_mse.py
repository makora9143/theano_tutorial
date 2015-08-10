#! /usr/bin/env python
# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from collections import OrderedDict

from logistic_mnist import load_data


class GeneralLayer(object):
    def __init__(self, n_in, n_out, activation=T.tanh, rng_seed=1234):
        self.rng = np.random.RandomState(rng_seed)

        self.n_in = n_in
        self.n_out = n_out

        self.activation = activation

        self.W = theano.shared(
            np.asarray(
                self.rng.uniform(low=-0.08, high=0.08, size=(n_in, n_out)),
                dtype=theano.config.floatX
            ),
            name='W'
        )

        self.b = theano.shared(
            np.asarray(
                np.zeros((n_out,)),
                dtype=theano.config.floatX
            ),
            name='b'
        )

        self.params = [self.W, self.b]

        self.L1_norm = abs(self.W).sum()
        self.L2_norm = abs(self.W ** 2).sum()

    def forward_prop(self, x):
        h = self.activation(T.dot(x, self.W) + self.b)
        self.h = h
        return h

def main():
    batch_size = 20

    mnist = fetch_mldata('MNIST original')
    mnist_x = mnist.data.astype('float32')/255.0
    mnist_y = mnist.target.astype('int32')
    train_x, valid_x, train_y, valid_y = train_test_split(mnist_x, mnist_y, test_size=0.2, random_state=42)
    new_train_y = []
    new_valid_y = []
    for i in train_y:
        tmp = np.zeros(10)
        tmp.put(i, 1)
        new_train_y.append(tmp)

    for i in valid_y:
        tmp = np.zeros(10)
        tmp.put(i, 1)
        new_valid_y.append(tmp)

    train_y = np.asarray(new_train_y, dtype='int32')
    valid_y = np.asarray(new_valid_y, dtype='int32')

    n_train_batches = train_x.shape[0] / batch_size
    n_valid_batches = valid_x.shape[0] / batch_size

    print '... building model'

    index = T.lscalar()
    x = T.matrix('x')
    t = T.imatrix('t')

    layers = [
        GeneralLayer(28 * 28, 500),
        GeneralLayer(500, 500),
        GeneralLayer(500, 500),
        GeneralLayer(500, 10, T.nnet.softmax)
    ]
    params = []

    for i, layer in enumerate(layers):
        params += layer.params
        if i == 0:
            layer_out = layer.forward_prop(x)
        else:
            layer_out = layer.forward_prop(layer_out)

    y = layers[-1].h

    L1_norm = sum([layer.L1_norm for layer in layers])
    L2_norm = sum([layer.L2_norm for layer in layers])

    # Negative log likelihood
    # cost = -T.mean(T.log(y)[T.arange(x.shape[0]), t]) + 0.000 * L1_norm + 0.0001 * L2_norm
    cost = T.mean((y[T.arange(x.shape[0])] - t[T.arange(x.shape[0])])** 2)
    # error = T.mean(T.neq(T.argmax(y, axis=1), t))

    # gparams = [T.grad(cost, param) for param in params]
    gparams = T.grad(cost, params)
    gmomentums = [
        theano.shared(np.asarray(
            np.zeros_like(param.get_value(borrow=True)),
            dtype=theano.config.floatX)
        )
        for param in params
    ]
    updates = OrderedDict()

    # learning_rate = 0.1
    learning_rate = 0.01
    momentum = np.float(0.9)

    for param, gparam, gmomentum in zip(params, gparams, gmomentums):
        updates[gmomentum] = momentum * gmomentum - learning_rate * gparam
        updates[param] = param + updates[gmomentum]
        # updates[param] = param - learning_rate * gparam
    print '... compiling'

    train = theano.function(
        inputs=[x, t],
        outputs=[cost],
        updates=updates,
    )

    valid = theano.function(
        inputs=[x, t],
        outputs=[cost],
    )
    print '... training'

    for epoch in xrange(500):
        train_x, train_y = shuffle(train_x, train_y)
        for minibatch_index in xrange(n_train_batches):
            start = minibatch_index * batch_size
            end = start + batch_size
            train(train_x[start:end], train_y[start:end])
            iter = epoch * n_train_batches + minibatch_index
            if (iter + 1) % n_train_batches == 0:
                print '... validating'
                this_validation_loss = valid(valid_x, valid_y)
                print this_validation_loss
                # print ("EPOCH:: %i, Validation cost: %f" % (epoch+1, this_validation_loss))

if __name__ == '__main__':
    main()




# End of Line.
