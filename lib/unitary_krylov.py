"""Unitary Krylov diagonalization studies."""
from functools import partial
import numpy as np
from scipy.linalg import eigvalsh
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.jax_experimental_sparse_linalg import lobpcg_standard


def make_hvec(hamiltonian, variable=False, dtype=np.complex128):
    nq = hamiltonian.num_qubits
    z_arr = np.array([1., -1.], dtype=dtype)
    zz_arr = np.array([[1., -1.], [-1., 1.]], dtype=dtype)
    shape_template = [1] * nq

    def hvec(state, plaquette_energy=None):
        shape_extra = state.shape[:-1]
        state = state.reshape(shape_extra + (2,) * nq)
        result = jnp.zeros_like(state)
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs.real):
            zs = np.nonzero(pauli.z[::-1])[0].astype(np.int64)
            xs = np.nonzero(pauli.x[::-1])[0].astype(np.int64)
            if zs.shape[0] == 1:
                shape = list(shape_template)
                shape[zs[0]] = 2
                result += jnp.reshape(coeff * z_arr, shape) * state
            elif zs.shape[0] == 2:
                shape = list(shape_template)
                shape[zs[0]] = 2
                shape[zs[1]] = 2
                result += jnp.reshape(coeff * zz_arr, shape) * state
            else:
                axis = len(shape_extra) + xs[0]
                index = (slice(None),) * axis
                index += (slice(None, None, -1),)
                if variable:
                    result += -plaquette_energy * state[index]
                else:
                    result += coeff * state[index]

        return result.reshape(shape_extra + (2 ** nq,))

    if variable:
        hvec = jax.jit(hvec)
    else:
        hvec = partial(jax.jit, static_argnums=[1])(hvec)

    return hvec


def make_trotter_uvec(hamiltonian, delta_t):
    # exp(-iHΔt)x
    nq = hamiltonian.num_qubits
    z_arr = np.array([1., -1.], dtype=np.complex128)
    zz_arr = np.array([[1., -1.], [-1., 1.]], dtype=np.complex128)
    shape_template = [1] * nq

    @jax.jit
    def apply_diag(result):
        """Apply exp(-i*0.5*Hdiag*Δt)."""
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            zs = np.nonzero(pauli.z[::-1])[0].astype(np.int64)
            if zs.shape[0] == 1:
                shape = list(shape_template)
                shape[zs[0]] = 2
                result *= jnp.reshape(np.exp(-0.5j * coeff * delta_t * z_arr), shape)
            elif zs.shape[0] == 2:
                shape = list(shape_template)
                shape[zs[0]] = 2
                shape[zs[1]] = 2
                result *= jnp.reshape(np.exp(-0.5j * coeff * delta_t * zz_arr), shape)
        return result

    @jax.jit
    def trotter_uvec(state):
        result = state.reshape((2,) * nq)
        result = apply_diag(result)

        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            xs = np.nonzero(pauli.x[::-1])[0].astype(np.int64)
            if xs.shape[0] == 1:
                rx_mat = np.array(
                    [
                        [np.cos(coeff * delta_t), -1.j * np.sin(coeff * delta_t)],
                        [-1.j * np.sin(coeff * delta_t), np.cos(coeff * delta_t)]
                    ]
                )
                result = jnp.moveaxis(jnp.tensordot(rx_mat, result, ([1], [xs[0]])), 0, xs[0])

        result = apply_diag(result)

        return result.reshape(-1)

    return trotter_uvec


@partial(jax.jit, static_argnums=[0, 1])
def integrate(hvec, nplaq, tpoints, psi0=None):
    if psi0 is None:
        psi0 = jax.nn.one_hot(0, 2 ** nplaq, dtype=np.complex128)
    return odeint(lambda state, _: -1.j * hvec(state), psi0, tpoints, rtol=1.e-10, atol=1.e-10)


@partial(jax.jit, static_argnums=[0, 1, 2, 3, 4])
def simulate(
    trotter_uvec,
    nplaq,
    krylov_dim,
    num_substeps=2,
    dims=None,
    psi0=None,
):
    if psi0 is None:
        psi0 = jax.nn.one_hot(0, 2 ** nplaq, dtype=np.complex128)

    def f(carry, _):
        y = carry
        for _ in range(num_substeps):
            y = trotter_uvec(y)
        return y, y

    psis = jax.lax.scan(f, psi0, length=krylov_dim)[1]
    if dims is None:
        return jnp.concatenate([psi0[None, :], psis], axis=0)

    dims = list(sorted(dims))
    if min(dims) != 0:
        indices = np.array(dims) - 1
        return psis[indices]

    psi_list = [psi0[None, :] if dim == 0 else psis[dim - 1] for dim in dims]
    return jnp.concatenate(psi_list, axis=0)



@partial(jax.jit, static_argnums=[0, 1, 2, 3])
def sample(trotter_uvec, nplaq, num_steps, shots, psi0=None):
    state = psi0
    if state is None:
        state = jax.nn.one_hot(0, 2 ** nplaq, dtype=np.complex128)

    for _ in range(num_steps):
        state = trotter_uvec(state)
    cumprobs = jnp.cumsum(jnp.square(jnp.abs(state)))

    key = jax.random.key(1234 + num_steps)
    xvals = jax.random.uniform(key, shots)
    return jnp.searchsorted(cumprobs, xvals)


def exact_diag(config, plaquette_energy):
    dual_lattice = TriangularZ2Lattice(config).plaquette_dual()
    hamiltonian = dual_lattice.make_hamiltonian(plaquette_energy)
    nplaq = dual_lattice.num_plaquettes
    hvec = make_hvec(hamiltonian)

    @jax.jit
    def compute(xmat):
        # pylint: disable-next=unbalanced-tuple-unpacking
        vals, vecs, _ = lobpcg_standard(lambda x: -hvec(x.T).T, xmat)
        return -vals, vecs

    xmat = np.zeros((2 ** nplaq, 2), dtype=np.complex128)
    xmat[0, 0] = 1.
    xmat[1 << np.arange(nplaq), 1] = 1. / np.sqrt(nplaq)
    return compute(xmat)


def compute_gen_eigvals(config, plaquette_energy, krylov_dim, delta_ts, num_substeps=2):
    gep_threshold = 1.e-10

    dual_lattice = TriangularZ2Lattice(config).plaquette_dual()
    nplaq = dual_lattice.num_plaquettes

    hamiltonian = dual_lattice.make_hamiltonian(plaquette_energy)
    hvec = make_hvec(hamiltonian)

    gen_eigvals = np.empty(delta_ts.shape + (krylov_dim + 1,))
    for idt, delta_t in enumerate(delta_ts):
        print(delta_t)

        @jax.jit
        def make_geneigval_problems():
            # pylint: disable-next=cell-var-from-loop
            trotter_uvec = make_trotter_uvec(hamiltonian, delta_t * 0.5)
            psi_sim = simulate(trotter_uvec, nplaq, krylov_dim, num_substeps=num_substeps)
            hpsi_sim = hvec(psi_sim)
            matrices = []
            for kdim in range(1, krylov_dim + 2):
                # krylov_matrix = jnp.tensordot(psi_sim[:kdim].conjugate(), hpsi_sim[:kdim],
                #                               [[1], [1]])
                # psi_matrix = jnp.tensordot(psi_sim[:kdim].conjugate(), psi_sim[:kdim], [[1], [1]])
                krylov_matrix = jnp.einsum('ij,kj->ik', psi_sim[:kdim].conjugate(), hpsi_sim[:kdim])
                psi_matrix = jnp.einsum('ij,kj->ik', psi_sim[:kdim].conjugate(), psi_sim[:kdim])
                psi_eigvals, psi_unitary = jnp.linalg.eigh(psi_matrix)
                start_dim = jnp.searchsorted(psi_eigvals, gep_threshold)
                psi_unitary_trunc = psi_unitary * jnp.asarray(psi_eigvals > gep_threshold)
                krylov_matrix = psi_unitary_trunc.conjugate().T @ krylov_matrix @ psi_unitary_trunc
                psi_matrix = psi_unitary_trunc.conjugate().T @ psi_matrix @ psi_unitary_trunc
                matrices.append((krylov_matrix, psi_matrix, start_dim))
            return matrices

        matrices = make_geneigval_problems()

        for idim, (krylov_matrix, psi_matrix, start_dim) in enumerate(matrices):
            gen_eigvals[idt, idim] = eigvalsh(krylov_matrix[start_dim:, start_dim:],
                                              psi_matrix[start_dim:, start_dim:])[0]

    return gen_eigvals
