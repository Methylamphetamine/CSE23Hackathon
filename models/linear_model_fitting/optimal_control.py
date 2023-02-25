# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 23:33:41 2023

@author: Matthias K. Hoffmann
"""

import casadi
import numpy as np
from rkHandler import discretizeODE, rk_step
import matplotlib.pyplot as plt

n_horizon = 6
delta_temp = 1

class square(casadi.Callback):
    def __init__(self, name, opts={}):
        casadi.Callback.__init__(self)
        self.construct(name, opts)

    def init(self):
        print('initialising object')

    def get_n_in(self): return 1

    def get_n_out(self): return 2

    def get_sparsity_in(self, n_in):
        return casadi.Sparsity.dense(1, 1)

    def get_sparsity_out(self, n_out):
        return casadi.Sparsity.dense(1, 1)

    def eval(self, arg):
        x = arg[0]
        f = x**2
        jac = 2*x
        return [jac, f]

    def has_jacobian(self): return True

    def get_jacobian(self, name, innames, smth, opts):
        x = casadi.MX.sym('x',1,1)
        return casadi.Function('jacobian', [x], 2*x)

ocp = casadi.Opti()

x0 = ocp.parameter(3,1)
xf = ocp.parameter(3,1)
xh = ocp.variable(3, n_horizon)
ocp.subject_to(ocp.bounded(-delta_temp, xf-xh[:,-1], delta_temp))

x = casadi.horzcat(x0, xh)

u = ocp.variable(2, n_horizon)

