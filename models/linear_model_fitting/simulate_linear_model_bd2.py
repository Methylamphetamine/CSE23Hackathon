import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from process_data import process_data
import matplotlib.pyplot as plt

n_rooms = 3
matrices = np.load("linear_model_bd2_matrices_small.npy", allow_pickle=True).item()

A = matrices["A"]
BT = matrices["BT"]
BH = matrices["BH"]
hH = matrices["hH"]
DE = matrices["DE"]
DH = matrices["DH"]

def get_linear_ode(A, BT, BH, hH, DE, DH, u_times, times):
    '''
    A  = Heat transfer matrix between rooms
    BT = House environment temperature interaction
    BH = House environment heat interaction
    hH = human heat per capita
    DE = electrical equipment effect
    DH = HVAC effect
    '''
    u_int = interp1d(times, u_times, kind='previous')
    def ode(t, x):
        '''
        t = current time
        x = state
        eT = environment temperature
        eH = environment heat
        hC = human count
        cE = electrical equipment heat transfer
        cH = hvac heating
        '''
        x = x.reshape((-1))
        u = u_int(t)
        eT = u[:1] # (outside temp + ground temp)
        eH = u[1].reshape(-1) # solar radiation
        hC = u[2:2+n_rooms] # human count
        cE = u[2+n_rooms:2+n_rooms*2]
        cH = u[2+n_rooms*2:]
        return A@x + BT@eT + BH@eH + hH*hC + DE@cE + DH@cH
    return ode

cols, data_pd = process_data("bld2.csv")
data = data_pd.to_numpy()

idx_start = 12000

x_data = data[idx_start:, [9, 10, 11]].T
u_data = np.concatenate((data[:, 0].reshape(-1,1), (data[:, 4]+data[:,5]).reshape(-1,1), data[:, [6,7,8]], data[:,[18,22,26,33,35,37]]), 1).T
u_data = u_data[:, idx_start:]
times = np.arange(0, x_data.shape[1]*600, 600)

odefun = get_linear_ode(A, BT, BH, hH, DE, DH, u_data, times)

t_span = (0, 24*60*60*10)
x0 = x_data[:, 0]
rtol = 1e-6
atol = 1e-8
sol_sim = solve_ivp(odefun, t_span, x0, t_eval=np.arange(t_span[0], t_span[1], 600), rtol=rtol, atol=atol, method='RK45')

plt.plot(sol_sim.t, sol_sim.y.T, label="linear")
plt.plot(np.arange(0,24*60*60*10,600), x_data[:,range(24*6*10)].T, label="real")
plt.legend()

