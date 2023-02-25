# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:06:00 2023

@author: Matthias K. Hoffmann
"""

import casadi
import numpy as np
from rkHandler import discretizeODE
import matplotlib.pyplot as plt

n_steps = 0
n_data  = 200
n_rooms = 3

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

opt = casadi.Opti()
alpha = np.ones((6,))#opt.variable(n_rooms)
# opt.subject_to(alpha >= 0)
A = casadi.MX(n_rooms, n_rooms)
alpha_counter = 0
for i in range(0,n_rooms):
    for j in range(i+1,n_rooms):
        A[i,j] = alpha[alpha_counter]
        A[j,i] = alpha[alpha_counter]
        alpha_counter += 1

A -= casadi.diag(casadi.sum2(A))
        
BT = np.ones((n_rooms, 2)) #opt.variable(n_rooms, 2) # 2: outside temperature + ground

A -= casadi.diag(casadi.sum2(BT)) # influence based on the external temperatures

BH = np.ones((n_rooms, 1))#opt.variable(n_rooms, 1) # solar radiation

hH = 1 #opt.variable(1) # human heat generation in a room

DE = casadi.diag(np.ones((n_rooms, 1)))#casadi.diag(opt.variable(n_rooms, 1)) # electricity heat
DH = casadi.diag(np.zeros((n_rooms, 1)))#casadi.diag(opt.variable(n_rooms, 1)) # hvac heating+cooling

odefun = get_linear_ode(A, BT, BH, hH, DE, DH)

# u_dim = (outside temp + ground temp) + solar radiation + humans per room + electric heat per room + hvac per room
u_dim = 2+1+n_rooms+n_rooms+n_rooms
dt = 0.1/(n_steps+1)
FRK4 = discretizeODE(odefun, n_rooms, u_dim, dt) 

x = opt.variable(n_rooms, n_data)
# x0 = np.array([[20, 30, 5]]).T
# u = np.concatenate((np.array([[30], [5], [-10], [1], [0], [0], [2], [0], [0]]), np.zeros( (u_dim-9,1) )), 0)

# for i in range(0, n_data-1):
#     opt.subject_to( x[:, i+1] == FRK4( i*10, x[:, i], u) )

x0 = np.array([[20, 30, 5]]).T
u = np.concatenate((np.array([[30], [5], [-10], [1], [0], [0], [2], [0], [0]]), np.zeros( (u_dim-9,1) )), 0)

for i in range(0, n_data):
    opt.subject_to( x[:, i] == FRK4( i*10, x_data[:, i], u_data[:,i]) )

opt.minimize()

opt.solver('ipopt')
sol = opt.solve()
plt.plot(sol.value(x.T))