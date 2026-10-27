"""Iterative SQD with perturbative core."""
import math
import logging
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from jax.sharding import get_abstract_mesh
from flax import nnx
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import run_sqd


def make_pt_core(
    num_bits: int,
    max_excitations: int,
    add_padding: bool = True,
    packed: bool = True
):
    """Return all bit combinations for the given number of total bits and maximum excitations.

    Algorithm:
    - Iteratively add bits
    - At iteration k, create an array of all combinations of k bits and another array of indices
      of the rightmost raised bit (endpoints)
    - In iteration k+1, expand the k-bit array according to the endpoints. For rows with endpoint m,
      raise bits m+1 to N-1.
    """
    if add_padding:
        eye = np.packbits(np.eye(num_bits + 1, dtype=np.uint8), axis=1)[1:]
    else:
        eye = np.packbits(np.eye(num_bits, dtype=np.uint8), axis=1)
    nbytes = eye.shape[1]
    cum_combs = np.cumsum([math.comb(num_bits, iq) for iq in range(max_excitations + 1)])
    all_states = np.zeros((cum_combs[-1], nbytes), dtype=np.uint8)

    states = np.zeros((1, nbytes), dtype=np.uint8)
    endpoints = np.array([-1])

    for nbits in range(1, max_excitations + 1):
        size = np.sum(num_bits - 1 - endpoints)
        states_e = np.empty((size, nbytes), dtype=np.uint8)
        endpoints_e = np.empty(size, dtype=np.int32)
        idx = 0
        for startpoint in range(nbits - 1, num_bits):
            rows = np.nonzero(endpoints == startpoint - 1)[0]
            nrows = rows.shape[0]
            size = nrows * (num_bits - startpoint)
            states_e[idx:idx + size] = (states[rows, None] + eye[startpoint:]).reshape((-1, nbytes))
            endpoints_e[idx:idx + size] = np.tile(np.arange(startpoint, num_bits), nrows)
            idx += size

        states = states_e
        endpoints = endpoints_e
        all_states[cum_combs[nbits - 1]:cum_combs[nbits]] = states

    if not packed:
        all_states = np.unpackbits(all_states, axis=1)[:, :num_bits]

    return all_states


@nnx.jit(static_argnames=['num_gen'])
def _generate_recovery(generators, syndrome_data, plaq_data, num_gen):
    @nnx.vmap
    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    def generate_fn(generator, syndrome_batch, plaq_batch):
        flips = generator.sample(syndrome_batch, num_gen)
        states = plaq_batch[None, :, :] ^ flips
        # Pad with an extra bit for sqd
        states = jnp.pad(states, [(0, 0), (0, 0), (1, 0)])
        return generator, states

    shape = (np.prod(plaq_data.shape[:3]) * num_gen, plaq_data.shape[3] + 1)
    gen_states = generate_fn(generators, syndrome_data, plaq_data)[1].reshape(shape)
    return jnp.packbits(gen_states, axis=1)


@nnx.jit(static_argnames=['return_best'])
def compose_and_diagonalize(
    hamiltonian: PauliSumXZ,
    gen_states: NDArray,
    pt_core: NDArray,
    carryover: NDArray,
    return_best: int = 0
):
    states_list = [gen_states, pt_core, carryover]
    if (not (mesh := get_abstract_mesh()).empty
            and (resid := sum(arr.shape[0] for arr in states_list) % mesh.size) != 0):
        states_list.append(jnp.zeros((resid, gen_states.shape[-1]), dtype=np.uint8))
    states_p = jnp.concatenate(states_list, axis=0)
    eigval, eigvec, states_u, subspace_dim = run_sqd(hamiltonian, states_p, states_p.shape[0], True)
    if return_best == 0:
        return eigval, subspace_dim

    sort_idx = jnp.argsort(jnp.square(eigvec), descending=True, dtype=np.int32)
    sort_idx = jnp.where(sort_idx < subspace_dim, sort_idx, -1)
    sorted_states = states_u.at[sort_idx].get(mode='fill', fill_value=0,
                                              wrap_negative_indices=False)
    pt_max = jnp.max(jnp.sum(jnp.bitwise_count(pt_core), axis=1))
    selection = jnp.sum(jnp.bitwise_count(sorted_states), axis=1) > pt_max
    select_idx = jnp.cumsum(
        jnp.bincount(
            jnp.cumsum(selection, dtype=np.int32),
            length=return_best
        ),
        dtype=np.int32
    )
    states = sorted_states[select_idx]
    return eigval, subspace_dim, states


@nnx.jit(static_argnames=['num_gen', 'carryover_size', 'num_iter', 'log_level'])
def iterative_sqd(
    hamiltonian: PauliSumXZ,
    generators: nnx.Module,
    syndrome_data: NDArray,
    plaq_data: NDArray,
    num_gen: int,
    pt_core: NDArray,
    carryover_size: int,
    num_iter: int,
    tol: float,
    log_level: int = logging.INFO
):
    def cond_fn(val):
        iteration, history = val[:2]
        not_converged = jax.lax.cond(
            iteration < 2,
            lambda: True,
            lambda: history[0][iteration - 2] - history[0][iteration - 1] > tol
        )
        return jnp.logical_and(iteration < num_iter, not_converged)

    def run_and_record(_iteration, _history, _carryover, _generators, return_states=True):
        gen_states = _generate_recovery(_generators, syndrome_data, plaq_data, num_gen)
        result = compose_and_diagonalize(hamiltonian, gen_states, pt_core, _carryover,
                                         return_best=_carryover.shape[0] if return_states else 0)
        eigval, subspace_dim = result[:2]
        if log_level <= logging.INFO:
            jax.debug.print('Iteration {}: energy {}', _iteration, eigval)
        _history = (_history[0].at[_iteration].set(eigval),
                    _history[1].at[_iteration].set(subspace_dim))
        if return_states:
            return _history, result[2]
        return _history
        
    def body_fn(val):
        _iteration, _history, _carryover, _generators = val
        _history, _carryover = run_and_record(_iteration, _history, _carryover, _generators)
        return (_iteration + 1, _history, _carryover, _generators)

    init = (
        0,
        (jnp.empty(num_iter + 1, dtype=np.float64), jnp.empty(num_iter + 1, dtype=np.int32)),
        jnp.zeros((carryover_size, pt_core.shape[-1]), dtype=np.uint8),
        generators
    )
    iteration, history, selected_states = nnx.while_loop(cond_fn, body_fn, init)[:3]

    return run_and_record(iteration, history, selected_states, generators, return_states=False)
