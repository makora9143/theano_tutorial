#! /usr/bin/env python
# -*- coding: utf-8 -*-


import theano.tensor as T
from theano import function


x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = function([x], s)

print logistic([[0, 1], [-1, -2]])


# End of Line.
