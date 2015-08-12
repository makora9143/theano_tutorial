#! /usr/bin/env python
# -*- coding: utf-8 -*-

import timeit
import os
try:
    import PIL.Image as Image
except:
    import Image

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
from logistic_mnist import load_data


class RestrictedBolzmanMachine(object):
    def __init__(
            self,
            input=None,
            n_visible=28*28,
            n_hidden=500,
            W=None,
            hbias=None,
            vbias=None,
            numpy_rng=None,
            theano_rng=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(initial_W, name='W', borrow=True)

        if hbias is None:
            hbias = theano.shared(
                value=np.zeros(n_hidden, dtype=theano.config.floatX),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            vbias = theano.shared(
                value=np.zeros(n_visible, dtype=theano.config.floatX),
                name='vbias',
                borrow=True
            )

        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.hbias, self.vbias]

    def propagate_up(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propagate_up(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1,
                                             p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propagate_down(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propagate_down(h0_sample)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1,
                                             p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def get_cost_updates(self, learning_rate=0.1, persistent=None, k=1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        (
            [
             pre_sigmoid_nvs,
             nv_means,
             nv_samples,
             pre_sigmoid_nhs,
             nh_means,
             nh_samples
            ],
            updates
        ) = theano.scan(
            fn=self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )

        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * T.cast(learning_rate,
                                                     dtype=theano.config.floatX)
        if persistent:
            updates[persistent] = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])
        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        xi = T.round(self.input)
        fe_xi = self.free_energy(xi)
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.free_energy(xi_flip)
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy


def main(learning_rate=0.1,
         training_epochs=15,
         dataset='mnist.pkl.gz',
         batch_size=20,
         n_chains=20,
         n_samples=10,
         output_folder='rbm_plots',
         n_hidden=500):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    persistent_chain = theano.shared(np.zeros((batch_size, n_hidden),
                                               dtype=theano.config.floatX),
                                     borrow=True)

    rbm = RestrictedBolzmanMachine(
        input=x,
        n_visible=28 * 28,
        n_hidden=n_hidden,
        numpy_rng=rng,
        theano_rng=theano_rng
    )

    cost, updates = rbm.get_cost_updates(learning_rate=learning_rate,
                                         persistent=persistent_chain,
                                         k=15)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    train_rbm = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()

    for epoch in xrange(training_epochs):
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, np.mean(mean_cost)

        plotting_start = timeit.default_timer()
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))

    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        np.asarray(
            test_set_x.get_value(borrow=True)[test_idx: test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )

    plot_every = 1000

    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        fn=rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every
    )

    updates.update({persistent_vis_chain: vis_samples[-1]})

    sample_fn = theano.function(
        inputs=[],
        outputs=[
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    image_data = np.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype='uint8'
    )

    for idx in xrange(n_samples):
        vis_mf, vis_sample = sample_fn()
        print '... plotting sample', idx
        image_data[29 * idx: 29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    image = Image.fromarray(image_data)
    image.save('sample.png')
    os.chdir('../')


if __name__ == '__main__':
    main()

# End of Line.
