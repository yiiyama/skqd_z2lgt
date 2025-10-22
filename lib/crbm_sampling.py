"""Offloading the sampling routine of CRBM to numba (CPU)."""
import numpy as np
from numba import njit


@njit
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


@njit
def vbinomial(vp):
    return (np.random.random(vp.shape) > vp).astype(vp.dtype)


@njit
def h_activation(model, u_states, v_states):
    weights_hu = model[1]
    weights_hv = model[2]
    bias_h = model[4]
    delta_e = u_states @ weights_hu.T + v_states @ weights_hv.T
    delta_e += bias_h[None, :]
    return sigmoid(delta_e)


@njit
def v_activation(model, u_states, h_states):
    weights_vu = model[0]
    weights_hv = model[2]
    bias_v = model[3]
    delta_e = u_states @ weights_vu.T + h_states @ weights_hv
    delta_e += bias_v[None, :]
    return sigmoid(delta_e)


@njit
def generate_v_states(model, u_states, v_state):
    ph = h_activation(model, u_states, v_state)
    h_states = vbinomial(ph)
    pv = v_activation(model, u_states, h_states)
    return vbinomial(pv)


@njit
def crbm_sample(model, u_states, size, therm_steps):
    weights_vu = model[0]
    bias_v = model[3]
    delta_e = u_states @ weights_vu.T + bias_v[None, :]
    pv = sigmoid(delta_e)
    v_states = vbinomial(pv)

    for _ in range(therm_steps):
        v_states = generate_v_states(model, u_states, v_states)

    out = np.empty((size,) + v_states.shape, dtype=np.uint8)
    for isamp in range(size):
        v_states = generate_v_states(model, u_states, v_states)
        out[isamp] = v_states

    return out

