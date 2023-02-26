
import numpy as np
import math
from functools import partial
import pickle
import casadi

# %%
#44, 256, 256, 1

# %%
def SIREN(x, params):
    weights1, weights2, weights3, mu_x, var_x, mu_t, var_t = params
    x = (x - casadi.DM(mu_x).reshape((1,-1))) / casadi.DM(var_x).reshape((1,-1))

    def forward(x, weights):

        weights = weights['params']

        for key in list(weights.keys())[:-1]:
            x = x@casadi.DM(weights[key]['kernel']) + casadi.DM(weights[key]['bias']).reshape((1,-1))
            x = casadi.sin(x)

        key = list(weights.keys())[-1]
        x = x@casadi.DM(weights[key]['kernel']) + casadi.DM(weights[key]['bias']).reshape((1,-1))




        return x


    
    outputs = []

    for wbs in [weights1, weights2, weights3]:
        outputs.append(forward(x, wbs))

    outputs = casadi.horzcat(*outputs)
    return outputs * casadi.DM(var_t).reshape((1,-1)) + casadi.DM(mu_t).reshape((1,-1))




# %%
# models = np.load('../model.pkl', allow_pickle=True)
models =  np.load('jax_pinn_model.npy', allow_pickle=True)

# %%
def get_nn_fun():
    forward_fn = partial(SIREN, params = models)
    return forward_fn

# x = np.ones((1, 44,))
# test = casadi.MX.ones(1,44)
# forward_fn(test)
# y = forward_fn(x)

# print(y)


