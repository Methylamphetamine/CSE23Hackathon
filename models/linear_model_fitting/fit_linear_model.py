# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:06:00 2023

@author: Matthias K. Hoffmann
"""

import casadi
import numpy as np
from rkHandler import discretizeODE

n_steps = 0
n_data  = 1000
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
dt = 600/(n_steps+1)
discretizeODE(odefun, n_rooms, 2+1+n_rooms+n_rooms+n_rooms, dt ) 


