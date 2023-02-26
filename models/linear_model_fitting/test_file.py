# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 00:23:01 2023

@author: Matthias K. Hoffmann
"""

from casadi import *

import casadi.casadi as cs


class MyCallback(cs.Callback):

    def __init__(self, name, opts={}):
        cs.Callback.__init__(self)
        self.construct(name, opts)

    def get_n_in(self): return 1

    def get_n_out(self): return 1
    
    def get_sparsity_in(self, i):
        return Sparsity.dense(1,1)

    def get_sparsity_out(self, i):
        return Sparsity.dense(1)
    
    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self, arg):
        print(arg)
        x = arg[0]
        f = x**2
        return [f]
    
    def has_jacobian(self):
        return True
    
    def get_jacobian(self, *args):
        x = casadi.MX.sym("x", 1)
        y = casadi.MX.sym("y", 1)
        return cs.Function("sm", [x,y], [2*x])
    
f = MyCallback("f")
opt = casadi.Opti()
x = opt.variable(1)
jacobian(f(x), x)
opt.solver('ipopt')
opt.minimize(f(x))

opt.solve()