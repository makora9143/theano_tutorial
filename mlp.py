#! /usr/bin/env python
# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import numpy as np


import os
import sys
import timeit


from logistic_mnist import LogisticRegression, load_data


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh):
        self.input = input

        self.n_in = n_in
        self.n_out = n_out

        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-0.08, high=0.08, size=(n_in, n_out)),
                dtype=theano.config.floatX
            ),
            name='W'
        )
        self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX), name='b')

        self.params = [self.W, self.b]

        self.activation = activation

        self.output = activation(T.dot(input, self.W) + self.b)


class MultiLayerPerceptron(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.inputToHidden = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh)

        self.hiddenToOutput = LogisticRegression(
            input=self.inputToHidden.output,
            n_in=n_hidden,
            n_out=n_out)

        self.L1_norm = (
            abs(self.inputToHidden.W).sum() +
            abs(self.hiddenToOutput.W).sum()
        )

        self.L2_norm = (
            abs(self.inputToHidden.W ** 2).sum() +
            abs(self.hiddenToOutput.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.hiddenToOutput.negative_log_likelihood
        )

        self.errors = self.hiddenToOutput.errors

        self.params = (
            self.inputToHidden.params +
            self.hiddenToOutput.params
        )

        self.input = input


def init_mlp(learning_rate, L1_reg, L2_reg, dataset, batch_size, n_hidden):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    validate_set_x, validate_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_validate_batches = validate_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    batches = (n_train_batches, n_validate_batches, n_test_batches)

    print '... building the model'

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = np.random.RandomState(1234)

    classifier = MultiLayerPerceptron(
        rng=rng,
        input=x,
        n_in=28*28,
        n_hidden=n_hidden,
        n_out=10)

    cost = (
        classifier.negative_log_likelihood(y) +
        L1_reg * classifier.L1_norm +
        L2_reg * classifier.L2_norm
    )

    test_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y)],
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y)],
        givens={
            x: validate_set_x[index * batch_size: (index + 1) * batch_size],
            y: validate_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
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

    return train_model, validate_model, test_model, batches


def train_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001,
             dataset='mnist.pkl.gz', n_epochs=500,
             batch_size=20, n_hidden=500):

    train_model, validate_model, test_model, batches = init_mlp(
        learning_rate=learning_rate,
        L1_reg=L1_reg,
        L2_reg=L2_reg,
        dataset=dataset,
        batch_size=batch_size,
        n_hidden=n_hidden
    )

    n_train_batches, n_valid_batches, n_test_batches = batches

    print '... training mlp'

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




if __name__ == '__main__':
    # init_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001,
    #          dataset='mnist.pkl.gz', batch_size=20, n_hidden=500)
    train_mlp()

# End of Line.
