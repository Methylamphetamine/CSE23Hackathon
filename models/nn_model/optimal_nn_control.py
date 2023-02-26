# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 23:33:41 2023

@author: Matthias K. Hoffmann
"""

import casadi
import numpy as np
from rkHandler import discretizeODE, rk_step
import matplotlib.pyplot as plt
from process_data import process_data

n_horizon = 40
w = 0.9999
delta_temp = 0
ocp = casadi.Opti()

cols, data_pd = process_data("bld1.csv")
data = data_pd.to_numpy()

idx_start = 4000

x_data = data[idx_start:, [10, 14, 16]].T
u_data = np.concatenate((data[:, [0, 4, 3, 6]], np.zeros((data.shape[0],2)),  data[:, 7].reshape((-1,1)), np.zeros((data.shape[0],2)),  (data[:, 8] - data[:,9]).reshape((-1, 1)), np.zeros((data.shape[0],2))), 1).T
u_data = u_data[:, idx_start:]

n_rooms = 3
dt = 600

odefun = get_linear_ode(A, BT, BH, hH, DE, DH)

x0 = ocp.parameter(3,1)
xf = ocp.parameter(3,1)
xh = ocp.variable(3, n_horizon)
ocp.subject_to(ocp.bounded(-delta_temp, xf[0]-xh[0,-1], delta_temp))

x = casadi.horzcat(x0, xh)
u = ocp.variable(3, n_horizon)

ocp.subject_to(ocp.bounded(-50000, u, 50000))

for i in range(n_horizon):
    u_in = casadi.vertcat(u_data[:-3,i].reshape((-1,1)), u[:,i].reshape((-1,1)))
    ocp.subject_to( x[:, i+1] == rk_step( odefun, i*600, x[:, i].reshape((-1,1)), u_in, dt) )

ocp.solver('ipopt')
ocp.set_value(x0, x_data[:,0])
ocp.set_value(xf, 22)
diff_temp = x-22
J1 = diff_temp.reshape((1,-1))@diff_temp.reshape((-1,1))
J2 = u.reshape((1,-1))@u.reshape((-1,1))
ocp.minimize(J1*w+J2*(1-w)*1e-2)
sol = ocp.solve()