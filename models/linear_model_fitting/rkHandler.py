# -*- coding: utf-8 -*-
"""
Functions to discretize the tube ode using runge-kutta schemes
"""

import casadi
import numpy as np

def discretizeODE(odefun, xdim, udim, dt, method="rk4"):
    butcher = getMethod(method)
    a = butcher[0:-1, 1:]
    b = butcher[-1, 1:]
    c = butcher[0:-1, 0]
    
    nStage = len(c)
    k = casadi.MX(xdim, nStage)
    x = casadi.MX.sym('x', xdim, 1)
    u = casadi.MX.sym('u', udim, 1)
    t = casadi.MX.sym('t', 1, 1)
    
    for iStage in range(nStage):
        k[:, iStage] = odefun(t + dt*c[iStage], x + dt * k @ a[iStage,:].reshape(nStage, 1), u)
        
    xplus = x + dt * k @ b.reshape(nStage,1)

    return casadi.Function('F', [t, x, u], [xplus])

def rk_step(odefun, t, x, u, dt, method="rk4"):
    butcher = getMethod(method)
    a = butcher[0:-1, 1:]
    b = butcher[-1, 1:]
    c = butcher[0:-1, 0]
    
    nStage = len(c)
    k = casadi.MX(x.shape[0], nStage)
    
    for iStage in range(nStage):
        k[:, iStage] = odefun(t + dt*c[iStage], x + dt * k @ a[iStage,:].reshape(nStage, 1), u)
        
    xplus = x + dt * k @ b.reshape(nStage,1)

    return xplus


def getMethod(method):
    if method == "rk1" or method == "euler":
        butcher = np.diag([0,1])
        
    elif method == "rk2":
        butcher  = np.diag([0, 1/2, 1]) + np.diag([1/2, 0], -1)
        
    elif method == "heun":
        butcher  = np.diag([0, 1, 1/2]) + np.diag([1, 1/2], -1)
        
    elif method == "rk3" or method == "simpson":
        butcher = np.diag([0, 1/2, 2, 1/6]) + np.diag([1/2, -1, 2/3], -1) + np.diag([1, 1/6], -2)
        
    elif method == "rk4":
        butcher = np.diag([0, 1/2, 1/2, 1, 1/6])
        butcher[1:4, 0] = np.array([1, 1, 2])/2
        butcher[-1, 1:4] = np.array([1, 2, 2])/6
        
    elif method == "3/8":
        butcher = np.diag([0, 1/3, 1, 1, 1/8]) + np.diag([1/3, -1/3, -1, 3/8], -1) + \
            np.diag([2/3, 1, 3/8], -2) + np.diag([1, 1/8], -3)
        
    return butcher
