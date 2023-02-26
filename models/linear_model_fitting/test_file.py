# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 00:23:01 2023

@author: Matthias K. Hoffmann
"""

from casadi import *

class testfunj(Callback):
    """Jacobian class for 'testfun'. Class-functions are overloaded from casadi"""
    def __init__(self, name, opts={}):
        Callback.__init__(self)
        self.construct(name, opts)

    def init(self):
        print('initialising object')

    def get_n_in(self): return 1

    def get_n_out(self): return 1

    def get_sparsity_in(self, n_in):
        return Sparsity.dense(1, 1)

    def get_sparsity_out(self, n_out):
        return Sparsity.dense(1, 1)

    def eval(self, arg):
        x = arg[0]
        f = x**2
        return f

    def has_jacobian(self): return True

    def get_jacobian(self, *args):
        J = Function("f", [x], [2*x])
        return J
    
f = testfunj("f")
f(2)
opt = casadi.Opti()
x = opt.variable(1)
jacobian(f(x), x)
opt.solver('ipopt')
opt.minimize(f(x))

opt.solve()