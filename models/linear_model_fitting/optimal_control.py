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
import random

random.seed(1000)

def ComputeOptimalControl(w=0.9999,delta_temp=0,theta_initial=[11,12,13],selected_month=7):

    n_horizon = 40
    #w = 0.9999
    #delta_temp = 0
    ocp = casadi.Opti()

    matrices = np.load("linear_model_matrices_small.npy", allow_pickle=True).item()

    cols, data_pd = process_data("bld1.csv")
    data = data_pd.to_numpy()

    print(cols)

    #idx_start = random.sample(range(4320),1)[0] + (selected_month-1) * 4320 
    idx_start = 4000

    x_data = data[idx_start:, [10, 14, 16]].T
    u_data = np.concatenate((data[:, [0, 4, 3, 6]], np.zeros((data.shape[0],2)),  data[:, 7].reshape((-1,1)), np.zeros((data.shape[0],2)),  (data[:, 8] - data[:,9]).reshape((-1, 1)), np.zeros((data.shape[0],2))), 1).T
    u_data = u_data[:, idx_start:]

    A = matrices["A"]
    BT = matrices["BT"]
    BH = matrices["BH"]
    hH = matrices["hH"]
    DE = matrices["DE"]
    DH = matrices["DH"]

    n_rooms = 3
    dt = 600

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
    theta_initial[0] = x_data[0,0]
    ocp.set_value(x0, np.array(theta_initial))
    theta_control = data[idx_start,12] # or 13
    ocp.set_value(xf, theta_control)
    diff_temp = x-theta_control
    J1 = diff_temp.reshape((1,-1))@diff_temp.reshape((-1,1))
    J2 = u.reshape((1,-1))@u.reshape((-1,1))
    ocp.minimize(J1*w+J2*(1-w)*1e-2)
    sol = ocp.solve()

    # Temperatures
    xval = sol.value(x)
    # Control
    uval = sol.value(u)

    N = xval.shape[1]
    #print(uval)

    plt.figure()
    plt.plot(xval[0,:])
    plt.plot(theta_control * np.ones((N,1)),'--')
    plt.plot(x_data[0,:N])
    plt.legend(["Temperature","Set-Point","Actual Operation"])
    plt.ylabel('Temperature')
    plt.xlabel('Time (10 mins/unit)')

    print(xval[0,:].shape)

    print(x_data.shape)

    plt.figure()
    actual_heating = data[idx_start:idx_start+N,8]-data[idx_start:idx_start+N,9]
    plt.plot(uval[0,:])
    plt.plot(actual_heating)
    plt.legend(["Heating Rate Control","Original Heating Rate"])
    plt.ylabel('Watt')
    plt.xlabel('Time (10 mins/unit)')
    # NO control points set for the following temperatures
    """ plt.figure()
    plt.plot(xval[1,:])

    plt.figure()
    plt.plot(xval[2,:]) """

    plt.show()

    np.savetxt("xval_b1.csv",xval)
    np.savetxt("uval_b1.csv",uval)

ComputeOptimalControl()