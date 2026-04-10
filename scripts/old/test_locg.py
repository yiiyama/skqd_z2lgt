import os
import logging
from pathlib import Path
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from skqd_z2lgt.ground_locg import ground_locg
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.sqd import get_hamiltonian_arrays, uniquify_states, get_diagonals, get_nondiagonals
from skqd_z2lgt.extensions import extensions
from skqd_z2lgt.tasks.common import make_dual_lattice
from skqd_z2lgt.tasks.preprocess import load_reco
from skqd_z2lgt.tasks.diagonalize import _prepare_data_and_models, _generate_cr

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.01'
jax.config.update('jax_enable_x64', True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

pkgpath = Path('/data/iiyama/2dz2/experiment_4x7_l0.5_ch9-13')

if os.path.isdir(pkgpath):
    print('Loading parameters from %s' % pkgpath)
    with open(pkgpath / 'parameters.json', 'r', encoding='utf-8') as src:
        parameters = Parameters.model_validate_json(src.read())

exp_data = load_reco(parameters, etype='exp')
dual_lattice = make_dual_lattice(parameters)
hamiltonian = dual_lattice.make_hamiltonian(parameters.lgt.plaquette_energy)
num_plaquettes = dual_lattice.num_plaquettes

crbm_models = _prepare_data_and_models(parameters, exp_data, logger)
ham_arrays = get_hamiltonian_arrays(hamiltonian, padding_bit=True, npmod=jnp)

states_list = _generate_cr(parameters, exp_data, crbm_models, logger)
states = np.concatenate(states_list, axis=0)
states = np.unpackbits(uniquify_states(states), axis=1)[:, :num_plaquettes]
states = extensions['perturbation_2q'](states, dual_lattice)

states_size = 200_000
pad_length = states_size - states.shape[0]
# Extend axis 1 by 1 bit for the padding flag
states = np.pad(states, [(0, 0), (1, 0)])
# Then extend axis 0 to states_size (fill with ones)
states = np.pad(states, [(0, pad_length), (0, 0)], constant_values=1)

paulis_d, coeffs_d, paulis_n, coeffs_n = ham_arrays

@jax.jit
def apply_pauli(iop, val):
    out, vec, idx_out, data = val
    out = out.at[idx_out[iop]].add(vec * data[iop], mode='drop', wrap_negative_indices=False)
    return (out, vec, idx_out, data)

@jax.jit
def apply_h_loop(vec, diagonals, idx_out, data):
    extra_dims = tuple(range(1, vec.ndim))
    diagonals = jnp.expand_dims(diagonals, extra_dims)
    result = vec * diagonals
    return jax.lax.fori_loop(0, idx_out.shape[0], apply_pauli, (result, vec, idx_out, data))[0]

jax.profiler.start_trace('/tmp/ground_locg')

states = uniquify_states(states, size=states.shape[0])
subspace_dim = jnp.searchsorted(states[:, 0] >> 7, 1)
diagonals = get_diagonals(paulis_d, coeffs_d, states)
rows, nondiag_data = get_nondiagonals(paulis_n, coeffs_n, states)
mask = jnp.logical_and(jnp.not_equal(rows, -1), jnp.less(rows, subspace_dim))
rows *= mask
nondiag_data *= mask
cols = jnp.tile(jnp.arange(states.shape[0]), rows.shape[0])
rows = rows.reshape(-1)
nondiag_data = nondiag_data.reshape(-1)

idx_in, idx_out, data = cols, rows, nondiag_data
vinit = jax.nn.one_hot(jnp.argmin(diagonals), diagonals.shape[0]).astype(np.complex128)

idx_out = idx_out.reshape((-1, diagonals.shape[0]))
data = data.reshape((-1, diagonals.shape[0]))
ground_locg(apply_h_loop, vinit, args=(diagonals, idx_out, data))
eigval_l, eigvec_l, iter_l = ground_locg(apply_h_loop, vinit, args=(diagonals, idx_out, data))

jax.profiler.stop_trace()

print(eigval_l)
