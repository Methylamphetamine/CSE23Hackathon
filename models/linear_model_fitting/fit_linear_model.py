# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:06:00 2023

@author: Matthias K. Hoffmann
"""

import casadi
import numpy as np
from rkHandler import discretizeODE, rk_step
import matplotlib.pyplot as plt
import sys
from process_data import process_data
import random

sys.path.append("..\\..")

n_data  = 1000
n_rooms = 3

cols, data_pd = process_data("bld1.csv")
data = data_pd.to_numpy()

# find indices at which the column names contain 'Zone Air Temperature'
# we want to make these what we're predicting.
# the rest is just inputs to our model
x_data = data[:, [10, 14, 16]].T
u_data = np.concatenate((data[:, [0, 4, 3, 6]], np.zeros((x_data.shape[1],2)),  data[:, 7].reshape((-1,1)), np.zeros((x_data.shape[1],2)),  (data[:, 8] - data[:,9]).reshape((-1, 1)), np.zeros((x_data.shape[1],2))), 1).T

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

def casadi_to_numpy(num):
    return np.array(casadi.DM(num))

opt = casadi.Opti()
alpha = opt.variable(n_rooms)
opt.subject_to(alpha >= 0)
A = casadi.MX(n_rooms, n_rooms)
alpha_counter = 0
for i in range(0,n_rooms):
    for j in range(i+1,n_rooms):
        A[i,j] = alpha[alpha_counter]
        A[j,i] = alpha[alpha_counter]
        alpha_counter += 1

A -= casadi.diag(casadi.sum2(A))
        
BT = opt.variable(n_rooms, 2) # 2: outside temperature + ground

A -= casadi.diag(casadi.sum2(BT)) # influence based on the external temperatures

BH = opt.variable(n_rooms, 1) # solar radiation

hH = opt.variable(1) # human heat generation in a room

DE = casadi.diag(opt.variable(n_rooms, 1)) # electricity heat
DH = casadi.diag(opt.variable(n_rooms, 1)) # hvac heating+cooling

odefun = get_linear_ode(A, BT, BH, hH, DE, DH)

# u_dim = (outside temp + ground temp) + solar radiation + humans per room + electric heat per room + hvac per room
u_dim = 2+1+n_rooms+n_rooms+n_rooms
dt = 10
# FRK4 = discretizeODE(odefun, n_rooms, u_dim, dt) 

x = opt.variable(n_rooms, n_data)
# x0 = np.array([[20, 30, 5]]).T
# u = np.concatenate((np.array([[30], [5], [-10], [1], [0], [0], [2], [0], [0]]), np.zeros( (u_dim-9,1) )), 0)

# for i in range(0, n_data-1):
#     opt.subject_to( x[:, i+1] == FRK4( i*10, x[:, i], u) )

idc = random.sample(range(x_data.shape[1]-1), n_data)

for i in range(n_data):
    idx = idc[i]
    opt.subject_to( x[:, i] == rk_step( odefun, i*600, x_data[:, idx].reshape((-1,1)), u_data[:,idx].reshape((-1,1)), dt) )

difference = x - x_data[:, np.array(idc) + 1]
opt.minimize(difference[:].reshape((1,-1)) @ difference[:].reshape((-1,1)))

opt.solver('ipopt')
sol = opt.solve()

matrix_dict = {"A": casadi_to_numpy(sol.value(A)), "BT": casadi_to_numpy(sol.value(BT)), "BH": casadi_to_numpy(sol.value(BH)),\
               "hH": casadi_to_numpy(sol.value(hH)), "DE": casadi_to_numpy(sol.value(DE)), "DH": casadi_to_numpy(sol.value(DH))}
#%%
np.save("linear_model_matrices_small", matrix_dict, allow_pickle=True)
