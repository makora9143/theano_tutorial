#! /usr/bin/env python
# -*- coding: utf-8 -*-


import cPickle
import gzip
import os
import sys
import timeit

import numpy as np
import theano
import theano.tensor as T


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.input = input

        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.params = [self.W, self.b]

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset='mnist.pkl.gz'):
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        print 'Downloading mnist data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(
            np.asarray(data_x, dtype=theano.config.floatX),
            borrow=borrow
        )
        shared_y = theano.shared(
            np.asarray(data_y, dtype=theano.config.floatX),
            borrow=borrow
        )
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [
        (train_set_x, train_set_y),
        (valid_set_x, valid_set_y),
        (test_set_x, test_set_y),
    ]
    return rval


def msgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                            dataset='mnist.pkl.gz', batch_size=600):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'

    index = T.lscalar('index')

    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)

    cost = classifier.negative_log_likelihood(y)

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [
        (classifier.W, classifier.W - learning_rate * g_W),
        (classifier.b, classifier.b - learning_rate * g_b)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )

    print '... training the model'

    patience = 5000
    patience_increase = 2

    improvement_threshold = 0.995

    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    test_score = 0
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1

        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                print '... now validating'
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print (
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    print '... now testing'

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print (
                        '    epoch %i, minibatch %i/%i, '
                        ' test error of best model %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )
            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    print (
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        ) % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)
    )


def main():
    msgd_optimization_mnist()


if __name__ == '__main__':
    main()


# End of Line.
