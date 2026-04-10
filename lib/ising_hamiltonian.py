"""Yet another Hv-op and Uv-op maker. Used in ground_4x8.py."""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec, AxisType


def get_coeffs(hamiltonian):
    z_coeffs = {}
    zz_coeffs = {}
    x_coeff = 0.

    # We know coeffs are all real
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs.real):
        zs = np.nonzero(pauli.z)[0].astype(np.int64)
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
            # We know there are only ZZ, Z, and X terms, and there is one X term per qubit
            x_coeff = coeff

    return z_coeffs, zz_coeffs, x_coeff


def get_shape_and_shardings(vec, qubit_partitioning):
    shape = tuple(2 ** np.array(qubit_partitioning))

    sharding = jax.typeof(vec).sharding
    if sharding.num_devices == 0:
        return shape, None, None

    in_spec = sharding.spec[0]  # ('X', 'Y', 'Z', ...)
    partitions = ()
    ipart = 0
    for nq in qubit_partitioning:
        partitions += (in_spec[ipart:ipart + nq],)
        ipart += nq
        if ipart >= len(in_spec):
            break
    return shape, sharding, NamedSharding(sharding.mesh, PartitionSpec(*partitions))


def make_apply_h(hamiltonian, axis_type=AxisType.Auto):
    """Apply-H highly optimized for the Ising Hamiltonian of the 2D Z2 LGT.

    Supply a NamedSharding compatible with a shape (2,) * nq array if sharding the input vector.
    """
    nq = hamiltonian.num_qubits

    def make_apply_z(qubit, coeff):
        def apply_z(vec):
            qpart = (nq - qubit - 1, 1, qubit)
            if axis_type == AxisType.Explicit:
                shape, out_sharding, tmp_sharding = get_shape_and_shardings(vec, qpart)
            else:
                shape = tuple(2 ** np.array(qpart))
                out_sharding, tmp_sharding = None, None

            op = jnp.array([coeff, -coeff]).reshape((1, 2, 1))
            vec = jnp.reshape(vec, shape, out_sharding=tmp_sharding)
            vec *= op
            vec = jnp.reshape(vec, (2 ** nq,), out_sharding=out_sharding)
            return vec

        return apply_z

    def make_apply_zz(qubit1, qubit2, coeff):
        def apply_zz(vec):
            qpart = (nq - qubit2 - 1, 1, qubit2 - qubit1 - 1, 1, qubit1)
            if axis_type == AxisType.Explicit:
                shape, out_sharding, tmp_sharding = get_shape_and_shardings(vec, qpart)
            else:
                shape = tuple(2 ** np.array(qpart))
                out_sharding, tmp_sharding = None, None

            op = jnp.array([[coeff, -coeff], [-coeff, coeff]]).reshape((1, 2, 1, 2, 1))
            vec = jnp.reshape(vec, shape, out_sharding=tmp_sharding)
            vec *= op
            vec = jnp.reshape(vec, (2 ** nq,), out_sharding=out_sharding)
            return vec

        return apply_zz

    def make_apply_x(qubit):
        def apply_x(vec):
            qpart = (nq - qubit - 1, 1, qubit)
            if axis_type == AxisType.Explicit:
                shape, out_sharding, tmp_sharding = get_shape_and_shardings(vec, qpart)
            else:
                shape = tuple(2 ** np.array(qpart))
                out_sharding, tmp_sharding = None, None

            vec = jnp.reshape(vec, shape, out_sharding=tmp_sharding)
            vec = jnp.flip(vec, axis=1)
            vec = jnp.reshape(vec, (2 ** nq,), out_sharding=out_sharding)
            return vec

        return apply_x

    z_coeffs, zz_coeffs, x_coeff = get_coeffs(hamiltonian)
    z_fns = [make_apply_z(q, c) for q, c in z_coeffs.items()]
    zz_fns = [make_apply_zz(q1, q2, c) for (q1, q2), c in zz_coeffs.items()]
    x_fns = [make_apply_x(q) for q in range(nq)]

    @jax.jit
    def apply_h(vec):
        result = jnp.zeros_like(vec)
        for fn in x_fns:
            result += fn(vec)
        result *= x_coeff
        for fn in z_fns + zz_fns:
            result += fn(vec)
        return result

    return apply_h


def make_apply_u(hamiltonian, axis_type=AxisType.Auto):
    """Apply-U highly optimized for the Ising Hamiltonian of the 2D Z2 LGT.

    Supply a NamedSharding compatible with a shape (2,) * nq array if sharding the input vector.
    """
    nq = hamiltonian.num_qubits

    def make_apply_rz(qubit, coeff):
        def apply_rz(vec, dt):
            # lattice.electric_evolution(dt) -> Rz(2 * coeff * dt) = exp(-1.j * coeff * dt * Z)
            qpart = (nq - qubit - 1, 1, qubit)
            if axis_type == AxisType.Explicit:
                shape, out_sharding, tmp_sharding = get_shape_and_shardings(vec, qpart)
            else:
                shape = tuple(2 ** np.array(qpart))
                out_sharding, tmp_sharding = None, None

            exponent = -1.j * coeff * dt
            op = jnp.exp(jnp.array([exponent, -exponent])).reshape((1, 2, 1))
            vec = jnp.reshape(vec, shape, out_sharding=tmp_sharding)
            vec *= op
            return jnp.reshape(vec, (2 ** nq,), out_sharding=out_sharding)

        return apply_rz

    def make_apply_rzz(qubit1, qubit2, coeff):
        def apply_rzz(vec, dt):
            # lattice.electric_evolution(dt) -> Rzz(2 * coeff * dt) = exp(-1.j * coeff * dt * ZZ)
            qpart = (nq - qubit2 - 1, 1, qubit2 - qubit1 - 1, 1, qubit1)
            if axis_type == AxisType.Explicit:
                shape, out_sharding, tmp_sharding = get_shape_and_shardings(vec, qpart)
            else:
                shape = tuple(2 ** np.array(qpart))
                out_sharding, tmp_sharding = None, None

            exponent = -1.j * coeff * dt
            op = jnp.exp(jnp.array([[exponent, -exponent], [-exponent, exponent]]))
            op = op.reshape((1, 2, 1, 2, 1))
            vec = jnp.reshape(vec, shape, out_sharding=tmp_sharding)
            vec *= op
            return jnp.reshape(vec, (2 ** nq,), out_sharding=out_sharding)

        return apply_rzz

    def make_apply_rx(qubit, coeff):
        def apply_rx(vec, dt):
            # lattice.magnetic_evolution(dt) -> Rx(2 * coeff * dt) = exp(-1.j * coeff * dt * X)
            qpart = (nq - qubit - 1, 1, qubit)
            if axis_type == AxisType.Explicit:
                shape, out_sharding, tmp_sharding = get_shape_and_shardings(vec, qpart)
            else:
                shape = tuple(2 ** np.array(qpart))
                out_sharding, tmp_sharding = None, None

            angle = coeff * dt
            op_d = jnp.cos(angle)
            op_n = -1.j * jnp.sin(angle)
            vec = jnp.reshape(vec, shape, out_sharding=tmp_sharding)
            vec = vec * op_d + jnp.flip(vec, axis=1) * op_n
            return jnp.reshape(vec, (2 ** nq,), out_sharding=out_sharding)

        return apply_rx

    z_coeffs, zz_coeffs, x_coeff = get_coeffs(hamiltonian)
    z_fns = [make_apply_rz(q, c) for q, c in z_coeffs.items()]
    zz_fns = [make_apply_rzz(axs[0], axs[1], c) for axs, c in zz_coeffs.items()]
    x_fns = [make_apply_rx(ax, x_coeff) for ax in range(nq)]

    @jax.jit
    def apply_u(vec, dt):
        for fn in z_fns + zz_fns:
            vec = fn(vec, 0.5 * dt)
        for fn in x_fns:
            vec = fn(vec, dt)
        for fn in z_fns + zz_fns:
            vec = fn(vec, 0.5 * dt)
        return vec

    return apply_u
