import os
import logging
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, AxisType, NamedSharding
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.ground_locg import ground_locg

jax.config.update('jax_enable_x64', True)
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger()

mesh_qubit = jax.make_mesh((2, 2, 2), ('X', 'Y', 'Z'), axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit))
sh_single = NamedSharding(mesh_qubit, PartitionSpec(('X', 'Y', 'Z')))
sh_qubit = NamedSharding(mesh_qubit, PartitionSpec('X', 'Y', 'Z'))

lattice = TriangularZ2Lattice((4, 8))
base_link_state = np.zeros(lattice.num_links, dtype=np.uint8)
base_link_state[::-1][[26, 28, 31, 34]] = 1
# base_link_state[::-1][[10, 13, 16]] = 1
# base_link_state[::-1][[20, 22, 25]] = 1
dual_lattice = lattice.plaquette_dual(base_link_state)
nplaq = lattice.num_plaquettes
hamiltonian = dual_lattice.make_hamiltonian(0.8)

def make_apply_z(nq, axis):
    def apply_z(vec, coeff):
        axes = list(range(nq))
        axes.remove(axis)
        return jnp.expand_dims(jnp.array([coeff, -coeff]), tuple(axes)) * vec

    return apply_z

def make_apply_zz(nq, axis1, axis2):
    def apply_zz(vec, coeff):
        axes = list(range(nq))
        axes.remove(axis1)
        axes.remove(axis2)
        return jnp.expand_dims(jnp.array([[coeff, -coeff], [-coeff, coeff]]), tuple(axes)) * vec

    return apply_zz

def make_apply_x(axis):
    def apply_x(vec):
        return jnp.flip(vec, axis=axis)

    return apply_x

def make_apply_h(hamiltonian):
    z_coeffs = {}
    zz_coeffs = {}
    x_coeff = 0.

    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs.real):
        zs = np.nonzero(pauli.z[::-1])[0].astype(np.int64)
        xs = np.nonzero(pauli.x[::-1])[0].astype(np.int64)
        if zs.shape[0] == 1:
            if zs[0] in z_coeffs:
                z_coeffs[zs[0]] += coeff
            else:
                z_coeffs[zs[0]] = coeff
        elif zs.shape[0] == 2:
            key = (zs[0], zs[1])
            if key in zz_coeffs:
                zz_coeffs[key] += coeff
            else:
                zz_coeffs[key] = coeff
        else:
            x_coeff = coeff

    nq = hamiltonian.num_qubits
    z_fns = [make_apply_z(nq, ax) for ax in z_coeffs.keys()]
    z_coeffs = jnp.array(list(z_coeffs.values()))
    zz_fns = [make_apply_zz(nq, *axs) for axs in zz_coeffs.keys()]
    zz_coeffs = jnp.array(list(zz_coeffs.values()))
    x_fns = [make_apply_x(ax) for ax in range(nq)]

    def z_body(iop, val):
        result, vec, coeffs = val
        result += jax.lax.switch(iop, z_fns, vec, coeffs[iop])
        return (result, vec, coeffs)

    def zz_body(iop, val):
        result, vec, coeffs = val
        result += jax.lax.switch(iop, zz_fns, vec, coeffs[iop])
        return (result, vec, coeffs)

    def x_body(iop, val):
        result, vec = val
        result += jax.lax.switch(iop, x_fns, vec)
        return (result, vec)


    @jax.jit
    def apply_h(vec):
        vec = vec.reshape((2,) * nplaq, out_sharding=sh_qubit)
        result = jnp.zeros_like(vec)
        result = jax.lax.fori_loop(0, nq, x_body, (result, vec))[0]
        result *= x_coeff
        result = jax.lax.fori_loop(0, z_coeffs.shape[0], z_body, (result, vec, z_coeffs))[0]
        result = jax.lax.fori_loop(0, zz_coeffs.shape[0], zz_body, (result, vec, zz_coeffs))[0]
        return result.reshape(-1, out_sharding=sh_single)

    return apply_h

apply_h = make_apply_h(hamiltonian)

eigval, eigvec, iter = ground_locg(apply_h, 0, vspace=(2 ** nplaq, np.float64), sharding=sh_single)
print(iter)
with h5py.File('/data/iiyama/ground_4x8.h5', 'w', libver='latest') as out:
    out.create_dataset('eigval', data=eigval)
    out.create_dataset('eigvec', data=eigvec)
# LOG.info('compiling')
# print(ground_locg(apply_h, 0, vspace=(2 ** nplaq, np.float64), sharding=sh_single)[0])
# LOG.info('tracing')
# with jax.profiler.trace('/tmp/ground_4x8'):
#     ground_locg(apply_h, 0, vspace=(2 ** nplaq, np.float64), sharding=sh_single)
# LOG.info('validation')
# print(jnp.linalg.eigvalsh(hamiltonian.to_matrix())[0])
