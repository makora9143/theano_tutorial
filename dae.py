#! /usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import timeit

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

import pylab

from logistic_mnist import load_data
from utils import tile_raster_images

import PIL.Image as Image


class DenoisingAutoencoder(object):
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 input=None,
                 n_visible=28*28,
                 n_hidden=500,
                 W=None,
                 bvis=None,
                 bhid=None,
                 tied=True
                 ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )


        self.b = bhid
        self.b_prime = bvis
        self.theano_rng = theano_rng

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.W = W
        if tied:
            self.W_prime = self.W.T
            self.params = [self.W, self.b, self.b_prime]
        else:
            initial_W_prime = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_hidden, n_visible)
                ),
                dtype=theano.config.floatX
            )

            self.W_prime = theano.shared(
                value=initial_W_prime,
                name='W_prime',
                borrow=True
            )

            self.params = [self.W, self.b, self.W_prime, self.b_prime]

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(
            size=input.shape,
            n=1,
            p=1 - corruption_level,
            dtype=theano.config.floatX
        ) * input

    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)

        z = self.get_reconstructed_input(y)

        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        cost = T.mean(L)

        gparams = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return (cost, updates)


def denosing():
    rng = np.random.RandomState(1234)
    input = T.fmatrix(name='input')
    corruption_level = T.fscalar(name='corruption_level')

    theano_rng = RandomStreams(rng.randint(2 ** 30))
    noise = theano_rng.binomial(
        size=input.shape,
        n=1,
        p=1 - corruption_level,
        dtype=theano.config.floatX
    )
    output = input * noise

    f = theano.function(
        inputs=[input, corruption_level],
        outputs=output,
        allow_input_downcast=True
    )

    from scipy.misc import imread
    img = imread('hayashi.jpg')
    img = img / 256.
    img = img[:, :, 0]

    pylab.gray()
    pylab.subplot(3, 5, 3)
    pylab.title('original')
    pylab.axis('off')
    pylab.imshow(img)
    for i in range(10):
        pylab.subplot(3, 5, 6+i)
        pylab.title("%d%% noise" % ((i+1)*10))
        pylab.axis('off')
        denoised = f(img, 0.1 * (i+1))
        pylab.imshow(denoised)
    pylab.show()


def main(learning_rate=0.1, training_epochs=15,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='dA_plots'):

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    index = T.lscalar()
    x = T.matrix(name='x')

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    dae = DenoisingAutoencoder(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28*28,
        n_hidden=500
    )

    cost, updates = dae.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate
    )

    train_autoencoder = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
        }
    )

    start_time = timeit.default_timer()

    for epoch in xrange(training_epochs):
        costs = []
        for batch_index in xrange(n_train_batches):
            costs.append(train_autoencoder(batch_index))

        print 'Training epoch %d, mean cost ' % epoch, np.mean(costs)

    end_time = timeit.default_timer()

    training_time = end_time - start_time
    print (
        'The 0% corruption code for file ',
        os.path.split(__file__)[1],
        ' ran for %.2fm' %(training_time / 60.)
    )

    image = Image.fromarray(
        tile_raster_images(
            X=dae.W.get_value(borrow=True).T,
            img_shape=(28, 28),
            tile_shape=(10, 10),
            tile_spacing=(1, 1)
        )
    )

    image.save('filters_corruption_0.png')

    dae = DenoisingAutoencoder(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28*28,
        n_hidden=500,
        tied=False
    )


    cost, updates = dae.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate
    )

    train_autoencoder = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
        }
    )

    start_time = timeit.default_timer()

    for epoch in xrange(training_epochs):
        costs = []
        for batch_index in xrange(n_train_batches):
            costs.append(train_autoencoder(batch_index))

        print 'Training epoch %d, mean cost ' % epoch, np.mean(costs)

    end_time = timeit.default_timer()

    training_time = end_time - start_time
    print (
        'The 30% corruption code for file ',
        os.path.split(__file__)[1],
        ' ran for %.2fm' %(training_time / 60.)
    )

    image = Image.fromarray(
        tile_raster_images(
            X=dae.W.get_value(borrow=True).T,
            img_shape=(28, 28),
            tile_shape=(10, 10),
            tile_spacing=(1, 1)
        )
    )

    image.save('filters_corruption_30.png')

    os.chdir('../')


if __name__ == '__main__':
    # main()
    denosing()

# End of Line.
