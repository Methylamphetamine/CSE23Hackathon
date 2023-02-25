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

ocp = casadi.Opti()

x0 = ocp.parameter(3,1)
xf = ocp.parameter(3,1)
xh = ocp.variable(3, n_horizon)
ocp.subject_to(ocp.bounded(-delta_temp, xf-x[:,-1], delta_temp))
x = casadi.horzcat(x0, xh)

u = ocp.variable()