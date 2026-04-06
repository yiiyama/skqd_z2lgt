"""Yet another Hv-op and Uv-op maker. Used in ground_4x8.py."""
import numpy as np
import jax
import jax.numpy as jnp


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


def make_apply_h(hamiltonian, sh_qubit=None):
    """Apply-H highly optimized for the Ising Hamiltonian of the 2D Z2 LGT.

    Supply a NamedSharding compatible with a shape (2,) * nq array if sharding the input vector.
    """
    nq = hamiltonian.num_qubits

    def make_apply_z(qubit, coeff):
        def apply_z(vec):
            shape = (2 ** (nq - qubit - 1), 2, 2 ** qubit)
            op = jnp.array([coeff, -coeff]).reshape((1, 2, 1))
            vec = jnp.reshape(vec, shape)
            vec *= op
            return jnp.reshape(vec, (2 ** nq,))

        return apply_z

    def make_apply_zz(qubit1, qubit2, coeff):
        def apply_zz(vec):
            shape = (2 ** (nq - qubit2 - 1), 2, 2 ** (qubit2 - qubit1 - 1), 2, 2 ** qubit1)
            op = jnp.array([[coeff, -coeff], [-coeff, coeff]]).reshape((1, 2, 1, 2, 1))
            vec = jnp.reshape(vec, shape)
            vec *= op
            return jnp.reshape(vec, (2 ** nq,))

        return apply_zz

    def make_apply_x(qubit):
        def apply_x(vec):
            shape = (2 ** (nq - qubit - 1), 2, 2 ** qubit)
            vec = jnp.reshape(vec, shape)
            vec = jnp.flip(vec, axis=1)
            return jnp.reshape(vec, (2 ** nq,))

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


def make_apply_u(hamiltonian, sh_qubit=None):
    """Apply-U highly optimized for the Ising Hamiltonian of the 2D Z2 LGT.

    Supply a NamedSharding compatible with a shape (2,) * nq array if sharding the input vector.
    """
    nq = hamiltonian.num_qubits

    def make_apply_rz(qubit, coeff):
        def apply_rz(vec, dt):
            # lattice.electric_evolution(dt) -> Rz(2 * coeff * dt) = exp(-1.j * coeff * dt * Z)
            shape = (2 ** (nq - qubit - 1), 2, 2 ** qubit)
            exponent = -1.j * coeff * dt
            op = jnp.exp(jnp.array([exponent, -exponent])).reshape((1, 2, 1))
            vec = jnp.reshape(vec, shape)
            vec *= op
            return jnp.reshape(vec, (2 ** nq,))

        return apply_rz

    def make_apply_rzz(qubit1, qubit2, coeff):
        def apply_rzz(vec, dt):
            # lattice.electric_evolution(dt) -> Rzz(2 * coeff * dt) = exp(-1.j * coeff * dt * ZZ)
            shape = (2 ** (nq - qubit2 - 1), 2, 2 ** (qubit2 - qubit1 - 1), 2, 2 ** qubit1)
            exponent = -1.j * coeff * dt
            op = jnp.exp(jnp.array([[exponent, -exponent], [-exponent, exponent]]))
            op = op.reshape((1, 2, 1, 2, 1))
            vec = jnp.reshape(vec, shape)
            vec *= op
            return jnp.reshape(vec, (2 ** nq,))

        return apply_rzz

    def make_apply_rx(qubit, coeff):
        def apply_rx(vec, dt):
            # lattice.magnetic_evolution(dt) -> Rx(2 * coeff * dt) = exp(-1.j * coeff * dt * X)
            shape = (2 ** (nq - qubit - 1), 2, 2 ** qubit)
            op_d = jnp.cos(coeff * dt)
            op_n = -1.j * jnp.sin(coeff * dt)
            vec = jnp.reshape(vec, shape)
            vec = vec * op_d + jnp.flip(vec, axis=1) * op_n
            return jnp.reshape(vec, (2 ** nq,))

        return apply_rx

    z_coeffs, zz_coeffs, x_coeff = get_coeffs(hamiltonian)
    z_fns = [make_apply_rz(q, c) for q, c in z_coeffs.items()]
    zz_fns = [make_apply_rzz(axs[0], axs[1], c) for axs, c in zz_coeffs.items()]
    x_fns = [make_apply_rx(ax, x_coeff) for ax in range(nq)]

    @jax.jit
    def apply_u(vec, dt):
        # sh_original = jax.typeof(vec).sharding
        # vec = vec.reshape((2,) * nq, out_sharding=sh_qubit)
        for fn in z_fns + zz_fns:
            vec = fn(vec, 0.5 * dt)
        for fn in x_fns:
            vec = fn(vec, dt)
        for fn in z_fns + zz_fns:
            vec = fn(vec, 0.5 * dt)
        # return vec.reshape(-1, out_sharding=sh_original)
        return vec

    return apply_u
