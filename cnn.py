#! /usr/bin/env python
# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
from theano.tensor.nnet import conv

import numpy as np

import pylab


rng = np.random.RandomState(1234)

input = T.tensor4(name='input')

w_shape = (2, 3, 60, 60)
w_bound = np.sqrt(3 * 60 * 60)
W = theano.shared(
    np.asanyarray(
        rng.uniform(
            low=-1.0 / w_bound,
            high=1.0 / w_bound,
            size=w_shape),
        dtype=input.dtype
    ),
    name='W'
)

b_shape = (2, )
b = theano.shared(
    np.asarray(
        rng.uniform(
            low=-0.5,
            high=0.5,
            size=b_shape),
        dtype=input.type
    ),
    name='b'
)

conv_out = conv.conv2d(input, W)

output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

f = theano.function([input], output)

from scipy.misc import imread

# 読み込むファイル名を指定
img = imread('hayashi.jpg')
img = img / 256.


img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, img.shape[0], img.shape[1]).astype('float32')
filtered_img = f(img_)
# filtered_img = g(filtered_img)

from theano.tensor.signal import downsample

input = T.dtensor4('input')

# Max Pooling のウィンドウサイズ
maxpool_shape = (2, 2)
# maxpool_shape = (2, 2)
pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)
h = theano.function([input],pool_out)

pool_img = h(filtered_img)



pylab.subplot(4, 3, 2)
pylab.title('original')
pylab.axis('off')
pylab.imshow(img)
pylab.gray();
pylab.subplot(4, 3, 4);
pylab.title('Feature Map: Red')
pylab.axis('off');
pylab.imshow(img_[0, 0, :, :])
pylab.subplot(4, 3, 5);
pylab.title('Feature Map: Green')
pylab.axis('off');
pylab.imshow(img_[0, 1, :, :])
pylab.subplot(4, 3, 6);
pylab.title('Feature Map: Blue')
pylab.axis('off');
pylab.imshow(img_[0, 2, :, :])
pylab.subplot(4, 3, 7);
pylab.title('Feature Map: 0')
pylab.axis('off');
pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(4, 3, 8);
pylab.title('Feature Map: 1')
pylab.axis('off');
pylab.imshow(filtered_img[0, 1, :, :])
pylab.subplot(4, 3, 10);
pylab.title('Feature Map: 1')
pylab.axis('off');
pylab.imshow(pool_img[0, 0, :, :])
pylab.subplot(4, 3, 11);
pylab.title('Feature Map: 1')
pylab.axis('off');
pylab.imshow(pool_img[0, 1, :, :])
pylab.show()
# End of Line.
