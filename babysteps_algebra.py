#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T
from theano import function


x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

print f(2, 3)
print f(16.3, 12.1)

x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
print f([[1, 2], [3, 4]], [[10, 20], [30, 40]])
print f(np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]]))

# exercise

print 'sample code'
a = T.vector()
out = a + a ** 10
f = function([a], out)
print f([0, 1, 2])

# modify and execute this code to compute this expression: a ** 2 + b ** 2 + 2 * a * b

print 'exercise code'
a = T.vector()
b = T.vector()
out = a ** 2 + b ** 2 + 2 * a * b
f = function([a, b], out)
print f([0, 1, 2], [3, 4, 5])


# End of Line.
