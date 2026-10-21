import numpy as np
from scipy.sparse.linalg import LinearOperator
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, NamedSharding
from qiskit.quantum_info import SparsePauliOp

SHARD_MIN_NQ = 26


def make_matvec(nq, zzops, zops, xops, npmod):
    def _matvec(x):
        vector = x.reshape((2,) * nq + x.shape[1:])
        result = npmod.zeros_like(vector)
        for (pos1, pos2), coeff in zzops:
            operand = npmod.moveaxis(vector, [pos1, pos2], [0, 1])
            if npmod is jnp:
                operand = operand.at[[0, 1], [1, 0]].multiply(-1.)
            else:
                operand[[0, 1], [1, 0]] *= -1.
            operand = npmod.moveaxis(operand, [0, 1], [pos1, pos2])
            result += operand * coeff
        for pos, coeff in zops:
            operand = npmod.moveaxis(vector, pos, 0)
            if npmod is jnp:
                operand = operand.at[1].multiply(-1.)
            else:
                operand[1] *= -1.
            operand = npmod.moveaxis(operand, 0, pos)
            result += operand * coeff
        for pos, coeff in xops:
            result += npmod.flip(vector, axis=pos) * coeff

        return result.reshape(x.shape)

    if npmod is jnp:
        _matvec = jax.jit(_matvec)
    return _matvec


class IsingHamiltonian(LinearOperator):
    def __init__(self, nq, zzops, zops, xops, npmod=np):
        super().__init__(np.complex128, (2 ** nq,) * 2)
        if npmod is jnp:
            matvec_pre = make_matvec(nq, zzops, zops, xops, jnp)
            if nq >= SHARD_MIN_NQ:
                mesh = jax.make_mesh((jax.device_count(),), ('dev',))

            def matvec_fn(x):
                if nq >= SHARD_MIN_NQ:
                    x = jax.device_put(x, NamedSharding(mesh, PartitionSpec('dev')))
                else:
                    x = jnp.array(x)
                return np.array(matvec_pre(x))

            self._matvec_fn = matvec_fn
        else:
            self._matvec_fn = make_matvec(nq, zzops, zops, xops, np)

    def _matvec(self, x):
        return self._matvec_fn(x)

    def _adjoint(self):
        return self


def ising_hamiltonian(hamiltonian: SparsePauliOp, npmod=np):
    """Construct a scipy sparse LinearOperator from a Hamiltonian containing diagonals and
    single-qubit Xs."""
    nq, zzops, zops, xops = parse_hamiltonian(hamiltonian)
    return IsingHamiltonian(nq, zzops, zops, xops, npmod=npmod)


def parse_hamiltonian(hamiltonian: SparsePauliOp):
    nq = hamiltonian.num_qubits
    zzops = []
    zops = []
    xops = []
    for term in hamiltonian:
        pauli = term.paulis[0]
        coeff = term.coeffs[0]
        zpos = np.nonzero(pauli.z)[0]
        if zpos.shape[0] == 2:
            zzops.append((nq - 1 - zpos, coeff))
        elif zpos.shape[0] == 1:
            zops.append((nq - 1 - zpos[0], coeff))
        else:
            xpos = np.nonzero(pauli.x)[0]
            xops.append((nq - 1 - xpos[0], coeff))

    return nq, zzops, zops, xops
