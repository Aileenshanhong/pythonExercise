# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 21:07:45 2016

@author: aileenlin
"""
import numpy
import theano.tensor as T
from theano import function
from theano import pp
from theano import In
from theano import shared

x = numpy.asarray([[1, 2], [3, 4], [5, 6]])
x.shape

x = T.dscalar()
y = T.dscalar()
w = T.dscalar()
z =( x + y)*w
g = 10
f = function([x, In(y, value = 1), In(w, value = 2, name = 'w_by_name')], z)
f(2,3, w_by_name=g)
numpy.allclose(f(16.3, 12.1), 28.4)
print(pp(z))

a = T.vector()
b = T.vector()
target = a ** 2 + b ** 2 + 2 * a * b
f1 = function([a, b], target)
print(f1([1, 2], [4, 5]))

x = T.dmatrix()
s = 1 / (1 + T.exp(-x))
logistic = function([x], s)
m = [[1, 2], [3, 4], [5, 6]]
logistic(m)

s2 = (1 + T.tanh(x/2))/2
logistic2 = function([x], s2)
logistic2([[1, 2], [3, 4], [5, 6]])

a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = function([a, b], [diff, abs_diff, diff_squared])
f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates = [(state, state+inc)])

print(state.get_value())
accumulator(300)
state.set_value(100)
accumulator(3)

decrementor = function([inc], state, updates = [(state, state-inc)])
decrementor(2)

new_state = shared(0)
new_accumulator = accumulator.copy(swap = {state: new_state})
new_accumulator(100)
print(new_state.get_value())

null_accumulator = accumulator.copy(delete_updates = True)

srng = T.shared_randomstreams.RandomStreams(seed = 234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates = True)
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)


state_after_v0 = rv_u.rng.get_value().get_state()
nearly_zeros()
v1 = f()
rng = rv_u.rng.get_value(borrow = True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow = True)


