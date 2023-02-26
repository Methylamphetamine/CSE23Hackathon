# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 23:33:41 2023

@author: Matthias K. Hoffmann
"""

import casadi
import numpy as np
import matplotlib.pyplot as plt
from rkHandler import discretizeODE, rk_step
from process_data import process_data
from jax_nn import get_nn_fun

n_horizon = 10
w = 1.
delta_temp = 1
ocp = casadi.Opti()

matrices = np.load("linear_model_matrices_small.npy", allow_pickle=True).item()

A = matrices["A"]
BT = matrices["BT"]
BH = matrices["BH"]
hH = matrices["hH"]
DE = matrices["DE"]
DH = matrices["DH"]

def get_linear_ode(A, BT, BH, hH, DE, DH):
    '''
    A  = Heat transfer matrix between rooms
    BT = House environment temperature interaction
    BH = House environment heat interaction
    hH = human heat per capita
    DE = electrical equipment effect
    DH = HVAC effect
    '''
    def ode(t, x, u):
        '''
        t = current time
        x = state
        eT = environment temperature
        eH = environment heat
        hC = human count
        cE = electrical equipment heat transfer
        cH = hvac heating
        '''
        # (outside temp + ground temp) + solar radiation + humans per room + electric heat per room + hvac per room
        eT = u[:2] # (outside temp + ground temp)
        eH = u[2] # solar radiation
        hC = u[3:3+n_rooms] # human count
        cE = u[3+n_rooms:3+n_rooms*2]
        cH = u[3+n_rooms*2:]
        return A@x + BT@eT + BH@eH + hH*hC + DE@cE + DH@cH
    return ode

odefun = get_linear_ode(A, BT, BH, hH, DE, DH)

cols, data_pd = process_data("bld1.csv")
data = data_pd.to_numpy()

idx_start = 0

x_data = data[idx_start:, [10, 12, 14]].T
u_data = np.concatenate((data[:, [0, 3, 4, 5, 6, 7]], (data[:, 8] - data[:, 9]).reshape((-1,1))), 1).T
u_data = u_data[:, idx_start:]

x_data_lin = data[idx_start:, [10, 12, 14]].T
u_data_lin = np.concatenate((data[:, [0, 4, 3, 6]], np.zeros((data.shape[0],2)),  data[:, 7].reshape((-1,1)), np.zeros((data.shape[0],2)),  (data[:, 8] - data[:,9]).reshape((-1, 1)), np.zeros((data.shape[0],2))), 1).T
u_data_lin = u_data_lin[:, idx_start:]

n_rooms = 3
dt = 600

onestep = get_nn_fun()

x0 = ocp.parameter(3,1)
xf = ocp.parameter(3,1)
xh = ocp.variable(3, n_horizon)
ocp.subject_to(ocp.bounded(-delta_temp, xf[0]-xh[0,-1], delta_temp))

x = casadi.horzcat(x0, xh)
u = ocp.variable(1, n_horizon)

ocp.subject_to(ocp.bounded(-6000, u, 6000))

for i in range(n_horizon):
    u_in = casadi.vertcat(u_data_lin[:-3,i].reshape((-1,1)), u[i].reshape((-1,1)), casadi.MX.zeros(2,1))
    x_pred = rk_step( odefun, i*600, x[:, i].reshape((-1,1)), u_in, dt)
    inputs = casadi.horzcat(u_data[0:6, i].reshape((1,-1)), u[i].reshape((1,-1)), x_data[:, i].reshape((1,-1)), x_pred.T)
    ocp.subject_to( x[:, i+1] == onestep(inputs).T )

ocp.solver('ipopt')
ocp.set_value(x0, x_data[:,0])
ocp.set_value(xf, 22)
diff_temp = x-22
J1 = diff_temp.reshape((1,-1))@diff_temp.reshape((-1,1))
J2 = u.reshape((1,-1))@u.reshape((-1,1))
ocp.minimize(J1*w+J2*(1-w))
sol = ocp.solve()

plt.figure()
plt.plot(sol.value(u))
plt.figure()
plt.plot(sol.value(x).T)