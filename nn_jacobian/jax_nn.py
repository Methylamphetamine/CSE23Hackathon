
import numpy as np
import math
from functools import partial
import pickle

# %%
#44, 256, 256, 1

# %%
def SIREN(x, params):
    weights1, weights2, weights3, mu_x, var_x, mu_t, var_t = params
    x = (x - mu_x) / var_x

    def forward(x, weights):

        weights = weights['params']

        for key in list(weights.keys())[:-1]:
            x = x@weights[key]['kernel'] + weights[key]['bias']
            x = np.sin(x)

        key = list(weights.keys())[-1]
        x = x@weights[key]['kernel'] + weights[key]['bias']




        return x


    
    outputs = []

    for wbs in [weights1, weights2, weights3]:
        outputs.append(forward(x, wbs))

    outputs = np.concatenate(outputs, -1)
    return outputs * var_t + mu_t




# %%
# models = np.load('../model.pkl', allow_pickle=True)
models = pickle.load(open('../model.pkl', 'rb'))

# %%
forward_fn = partial(SIREN, params = models)
x = np.ones((1, 44,))
y = forward_fn(x)

print(y)


