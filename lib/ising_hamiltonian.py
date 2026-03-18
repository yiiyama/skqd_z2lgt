"""Yet another Hv-op and Uv-op maker. Used in ground_4x8.py."""
import numpy as np
import jax
import jax.numpy as jnp


def make_apply_h(hamiltonian, sh_qubit=None):
    """Apply-H highly optimized for the Ising Hamiltonian of the 2D Z2 LGT.

    Supply a NamedSharding compatible with a shape (2,) * nq array if sharding the input vector.
    """
    z_coeffs = {}
    zz_coeffs = {}
    x_coeff = 0.

    # We know coeffs are all real
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs.real):
        zs = np.nonzero(pauli.z[::-1])[0].astype(np.int64)
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

    nq = hamiltonian.num_qubits

    def make_apply_z(axis):
        def apply_z(vec, coeff):
            axes = list(range(nq))
            axes.remove(axis)
            return jnp.expand_dims(jnp.array([coeff, -coeff]), tuple(axes)) * vec

        return apply_z

    def make_apply_zz(axis1, axis2):
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

    z_fns = [make_apply_z(ax) for ax in z_coeffs.keys()]
    z_coeffs = jnp.array(list(z_coeffs.values()))
    zz_fns = [make_apply_zz(*axs) for axs in zz_coeffs.keys()]
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
        vec = vec.reshape((2,) * nq, out_sharding=sh_qubit)
        result = jnp.zeros_like(vec)
        result = jax.lax.fori_loop(0, nq, x_body, (result, vec))[0]
        result *= x_coeff
        result = jax.lax.fori_loop(0, z_coeffs.shape[0], z_body, (result, vec, z_coeffs))[0]
        result = jax.lax.fori_loop(0, zz_coeffs.shape[0], zz_body, (result, vec, zz_coeffs))[0]
        return result.reshape(-1, out_sharding=jax.typeof(vec).sharding)

    return apply_h


def make_apply_u(hamiltonian, sh_qubit=None):
    """Apply-U highly optimized for the Ising Hamiltonian of the 2D Z2 LGT.

    Supply a NamedSharding compatible with a shape (2,) * nq array if sharding the input vector.
    """
    z_coeffs = {}
    zz_coeffs = {}
    plaquette_energy = 0.

    # We know coeffs are all real
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs.real):
        zs = np.nonzero(pauli.z[::-1])[0].astype(np.int64)
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

    nq = hamiltonian.num_qubits

    def make_apply_rz(axis, coeff):
        def apply_rz(vec, dt):
            # lattice.electric_evolution(dt) -> Rz(2 * coeff * dt) = exp(-1.j * coeff * dt * Z)
            axes = list(range(nq))
            axes.remove(axis)
            exponent = -1.j * coeff * dt
            pp = jnp.exp(exponent)
            pm = jnp.exp(-exponent)
            return jnp.expand_dims(jnp.array([pp, pm]), tuple(axes)) * vec

        return apply_rz

    def make_apply_rzz(axis1, axis2, coeff):
        def apply_rzz(vec, dt):
            # lattice.electric_evolution(dt) -> Rzz(2 * coeff * dt) = exp(-1.j * coeff * dt * ZZ)
            axes = list(range(nq))
            axes.remove(axis1)
            axes.remove(axis2)
            exponent = -1.j * coeff * dt
            pp = jnp.exp(exponent)
            pm = jnp.exp(-exponent)
            return jnp.expand_dims(jnp.array([[pp, pm], [pm, pp]]), tuple(axes)) * vec

        return apply_rzz

    def make_apply_rx(axis):
        def apply_rx(vec, dt):
            # lattice.magnetic_evolution(dt) -> Rx(-2 * lambda * dt) = exp(1.j * lambda * dt * X)
            angle = plaquette_energy * dt
            di = jnp.cos(angle)
            od = 1.j * jnp.sin(angle)
            return di * vec + od * jnp.flip(vec, axis=axis)

        return apply_rx

    z_fns = [make_apply_rz(ax, c) for ax, c in z_coeffs.items()]
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
        sh_original = jax.typeof(vec).sharding
        vec = vec.reshape((2,) * nq, out_sharding=sh_qubit)
        if len(z_fns):
            vec = jax.lax.fori_loop(0, len(z_fns), z_body, (vec, 0.5 * dt))[0]
        if len(zz_fns):
            vec = jax.lax.fori_loop(0, len(zz_fns), zz_body, (vec, 0.5 * dt))[0]
        vec = jax.lax.fori_loop(0, len(x_fns), x_body, (vec, dt))[0]
        if len(z_fns):
            vec = jax.lax.fori_loop(0, len(z_fns), z_body, (vec, 0.5 * dt))[0]
        if len(zz_fns):
            vec = jax.lax.fori_loop(0, len(zz_fns), zz_body, (vec, 0.5 * dt))[0]
        return vec.reshape(-1, out_sharding=sh_original)

    return apply_u
