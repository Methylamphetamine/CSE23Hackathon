# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 23:33:41 2023

@author: Matthias K. Hoffmann
"""

import casadi
import numpy as np
import matplotlib.pyplot as plt
from process_data import process_data
from jax_nn import get_nn_fun

n_horizon = 40
w = 0.9999
delta_temp = 1
ocp = casadi.Opti()

cols, data_pd = process_data("bld1.csv")
data = data_pd.to_numpy()

idx_start = 30000

x_data = data[idx_start:, [10, 12, 14]].T
u_data = np.concatenate((data[:, [0, 3, 4, 5, 6, 7]], (data[:, 8] - data[:, 9]).reshape((-1,1))), 1).T
u_data = u_data[:, idx_start:]

n_rooms = 3
dt = 600

onestep = get_nn_fun()

x0 = ocp.parameter(3,1)
xf = ocp.parameter(3,1)
xh = ocp.variable(3, n_horizon)
# ocp.subject_to(ocp.bounded(-delta_temp, xf[0]-xh[0,-1], delta_temp))

x = casadi.horzcat(x0, xh)
u = ocp.variable(1, n_horizon)

ocp.subject_to(ocp.bounded(-5000, u, 5000))

for i in range(n_horizon):
    inputs = casadi.horzcat(u_data[0:6, i].reshape((1,-1)), u[i].reshape((1,-1)), x_data[:, i].reshape((1,-1)))
    ocp.subject_to( x[:, i+1] == onestep(inputs).T )

ocp.solver('ipopt')
ocp.set_value(x0, x_data[:,0])
ocp.set_value(xf, 22)
diff_temp = x-22
J1 = diff_temp.reshape((1,-1))@diff_temp.reshape((-1,1))
J2 = u.reshape((1,-1))@u.reshape((-1,1))
ocp.minimize(J1)
sol = ocp.solve()

plt.plot(sol.value(u))