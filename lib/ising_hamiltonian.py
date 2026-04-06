"""Yet another Hv-op and Uv-op maker. Used in ground_4x8.py."""
import numpy as np
import jax
import jax.numpy as jnp


def get_coeffs(hamiltonian):
    z_coeffs = {}
    zz_coeffs = {}
    plaquette_energy = 0.

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
            plaquette_energy = -coeff

    return z_coeffs, zz_coeffs, plaquette_energy


def make_apply_h(hamiltonian, sh_qubit=None):
    """Apply-H highly optimized for the Ising Hamiltonian of the 2D Z2 LGT.

    Supply a NamedSharding compatible with a shape (2,) * nq array if sharding the input vector.
    """
    z_coeffs, zz_coeffs, plaquette_energy = get_coeffs(hamiltonian)
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

    z_fns = [make_apply_z(q, c) for q, c in z_coeffs.items()]
    zz_fns = [make_apply_zz(q1, q2, c) for (q1, q2), c in zz_coeffs.items()]
    x_fns = [make_apply_x(q) for q in range(nq)]

    def z_body(iop, val):
        result, vec = val
        result += jax.lax.switch(iop, z_fns, vec)
        return (result, vec)

    def zz_body(iop, val):
        result, vec = val
        result += jax.lax.switch(iop, zz_fns, vec)
        return (result, vec)

    def x_body(iop, val):
        result, vec = val
        result += jax.lax.switch(iop, x_fns, vec)
        return (result, vec)

    @jax.jit
    def apply_h(vec):
        result = jnp.zeros_like(vec)
        result = jax.lax.fori_loop(0, nq, x_body, (result, vec))[0]
        result *= -plaquette_energy
        if len(z_fns):
            result = jax.lax.fori_loop(0, len(z_fns), z_body, (result, vec))[0]
        if len(zz_fns):
            result = jax.lax.fori_loop(0, len(zz_fns), zz_body, (result, vec))[0]
        return result

    return apply_h


def make_apply_u(hamiltonian, sh_qubit=None):
    """Apply-U highly optimized for the Ising Hamiltonian of the 2D Z2 LGT.

    Supply a NamedSharding compatible with a shape (2,) * nq array if sharding the input vector.
    """
    z_coeffs, zz_coeffs, plaquette_energy = get_coeffs(hamiltonian)
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

    def make_apply_rx(qubit):
        def apply_rx(vec, dt):
            # lattice.magnetic_evolution(dt) -> Rx(-2 * lambda * dt) = exp(1.j * lambda * dt * X)
            shape = (2 ** (nq - qubit - 1), 2, 2 ** qubit)
            angle = plaquette_energy * dt
            op_d = jnp.cos(angle)
            op_n = 1.j * jnp.sin(angle)
            vec = jnp.reshape(vec, shape)
            vec = vec * op_d + jnp.flip(vec, axis=1) * op_n
            return jnp.reshape(vec, (2 ** nq,))

        return apply_rx

    z_fns = [make_apply_rz(q, c) for q, c in z_coeffs.items()]
    zz_fns = [make_apply_rzz(axs[0], axs[1], c) for axs, c in zz_coeffs.items()]
    x_fns = [make_apply_rx(ax) for ax in range(nq)]

    def z_body(iop, val):
        vec, dt = val
        vec = jax.lax.switch(iop, z_fns, vec, dt)
        return (vec, dt)

    def zz_body(iop, val):
        vec, dt = val
        vec = jax.lax.switch(iop, zz_fns, vec, dt)
        return (vec, dt)

    def x_body(iop, val):
        vec, dt = val
        vec = jax.lax.switch(iop, x_fns, vec, dt)
        return (vec, dt)

    @jax.jit
    def apply_u(vec, dt):
        # sh_original = jax.typeof(vec).sharding
        # vec = vec.reshape((2,) * nq, out_sharding=sh_qubit)
        if len(z_fns):
            vec = jax.lax.fori_loop(0, len(z_fns), z_body, (vec, 0.5 * dt))[0]
        if len(zz_fns):
            vec = jax.lax.fori_loop(0, len(zz_fns), zz_body, (vec, 0.5 * dt))[0]
        vec = jax.lax.fori_loop(0, len(x_fns), x_body, (vec, dt))[0]
        if len(z_fns):
            vec = jax.lax.fori_loop(0, len(z_fns), z_body, (vec, 0.5 * dt))[0]
        if len(zz_fns):
            vec = jax.lax.fori_loop(0, len(zz_fns), zz_body, (vec, 0.5 * dt))[0]
        # return vec.reshape(-1, out_sharding=sh_original)
        return vec

    return apply_u
