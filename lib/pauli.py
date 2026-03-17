# pylint: disable=cell-var-from-loop
"""Conversion of Pauli ops to JAX arrays. Migrated from qii-miyabi-kawasaki skqd_z2lgt for archival."""
from typing import Optional
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax._src.numpy import array_creation
from jax._src.lax import lax
from qiskit.quantum_info import SparsePauliOp


def op_to_arrays(op: SparsePauliOp) -> tuple[jax.Array, jax.Array]:
    """Convert Pauli strings into an array of {0,1,2,3} indices.

    Args:
        op: Sum of Pauli strings.

    Returns:
        Arrays of Pauli indices (shape [num_terms, num_qubits]) and coefficients ([num_terms]).
    """
    pauli_index = {c: i for i, c in enumerate('IXYZ')}
    index_array = jnp.array([[pauli_index[c] for c in p.to_label()] for p in op.paulis],
                            dtype=np.uint8)
    coeff_array = jnp.array(op.coeffs)
    return index_array, coeff_array


@jax.jit
def multi_pauli_map(
    pauli_strings: jax.Array,
    states: Optional[jax.Array] = None
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map computational basis states through multiple Pauli strings.

    Args:
        pauli_strings: Array of Pauli strings. Shape [..., num_qubits] with values in {0, 1, 2, 3}.
        states: Computational basis states (packed bitstrings) to map through the Pauli strings. If
            absent, all basis states for the given number of qubits are mapped.

    Returns:
        Indices of the mapped states (shape [..., subspace_dim]), signs of the corresponding matrix
        elements (shape [..., subspace_dim]), and the parity of the number of Ys in the Pauli
        strings (shape [prod(...)]).
    """
    if states is None:
        rows, signs = _v_pauli_map(pauli_strings)
        subspace_dim = 2 ** pauli_strings.shape[-1]
    else:
        match pauli_strings.ndim:
            case 2:
                rows, signs = _v_subspace_pauli_map(pauli_strings, states)
            case 3:
                rows, signs = _sv_subspace_pauli_map(pauli_strings, states)
            case _:
                raise ValueError('Too many dimensions in pauli_strings')

        subspace_dim = states.shape[0]

    # THIS IS INCOMPLETE - we need to count the number of 2s *mod 4* and book-keep the sign
    imaginary = (jnp.sum(jnp.equal(pauli_strings, 2), axis=-1) % 2).astype(np.uint8)

    shape = (-1, subspace_dim)
    # pylint: disable-next=possibly-used-before-assignment
    return rows.reshape(shape), signs.reshape(shape), imaginary.reshape(-1)


@jax.jit
def _pauli_map(pauli_string: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the row indices and matrix element phases of nonzero entries of the Pauli unitary.

    Phases are integers in units of pi/2.

    This method only works for num_qubits < 64, but the practical limit for enumerating all rows is
    much lower.

    Args:
        pauli_string: Array of {0,1,2,3} pauli indices, shape [num_qubits].

    Returns:
        Row indices (shape [2 ** num_qubits]) and matrix element signs ([2 ** num_qubits]) of the
        nonzero entries of the Pauli unitary, ordered by columns.
    """
    return jax.lax.cond(
        jnp.all(jnp.equal(pauli_string % 3, 0)),
        _pauli_map_diagonal,
        _pauli_map_nondiagonal,
        pauli_string
    )


_v_pauli_map = jax.jit(jax.vmap(_pauli_map))


@jax.jit
def _pauli_map_diagonal(pauli_string: jax.Array) -> tuple[jax.Array, jax.Array]:
    num_qubits = pauli_string.shape[0]
    rows = jnp.arange(2 ** num_qubits, dtype=np.int32)
    signs = jnp.zeros((2,) * num_qubits, dtype=np.uint8)

    for iq in range(num_qubits):
        signs = jax.lax.cond(
            jnp.equal(pauli_string[iq], 3),
            lambda: jnp.moveaxis(jnp.moveaxis(signs, iq, 0).at[1].add(1), 0, iq),
            lambda: signs
        )
    signs %= 2
    return rows, signs.reshape(-1)


@jax.jit
def _pauli_map_nondiagonal(pauli_string: jax.Array) -> tuple[jax.Array, jax.Array]:
    num_qubits = pauli_string.shape[0]
    rows = jnp.arange(2 ** num_qubits, dtype=np.int32).reshape((2,) * num_qubits)
    signs = jnp.zeros((2,) * num_qubits, dtype=np.uint8)

    for iq in range(num_qubits):
        rows = jax.lax.cond(
            jnp.not_equal(pauli_string[iq] % 3, 0),
            lambda: jnp.flip(rows, axis=iq),
            lambda: rows
        )
        signs = jax.lax.cond(
            jnp.greater(pauli_string[iq], 1),
            lambda: jnp.moveaxis(jnp.moveaxis(signs, iq, 0).at[1].add(1), 0, iq),
            lambda: signs
        )
    signs %= 2
    return rows.reshape(-1), signs.reshape(-1)


@jax.jit
def _subspace_pauli_map(
    pauli_string: jax.Array,
    states: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return the row numbers and matrix elements of the Pauli string for the given columns.

    Args:
        pauli_string: Shape [num_qubits]
        states: Packed computational basis states. Shape [subspace_dim, ceil(num_qubits / 8)]
    """
    return jax.lax.cond(
        jnp.all(jnp.equal(pauli_string % 3, 0)),
        lambda: _subspace_pauli_map_diagonal(pauli_string, states),
        lambda: _subspace_pauli_map_nondiagonal(pauli_string, states)
    )


_v_subspace_pauli_map = jax.jit(jax.vmap(_subspace_pauli_map, in_axes=(0, None)))


@jax.jit
def _sv_subspace_pauli_map(
    pauli_strings: jax.Array,
    states: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Temporally vectorized rows_and_elements. Cannot fullly vmap because of memory footprint."""
    def body_fn(_states, pauli_string_block):
        rows, signs = _v_subspace_pauli_map(pauli_string_block, _states)
        return _states, (rows, signs)

    return jax.lax.scan(
        body_fn, states, pauli_strings
    )[1]


@jax.jit
def _subspace_pauli_map_diagonal(
    pauli_string: jax.Array,
    states: jax.Array
) -> tuple[jax.Array, jax.Array]:
    is_signed = jnp.packbits(jnp.greater(pauli_string, 1).astype(np.uint8))
    rows = jnp.arange(np.prod(states.shape[:-1]), dtype=np.int32).reshape(states.shape[:-1])
    signs = jnp.sum(jnp.bitwise_count(states & is_signed), axis=-1) % 2
    return rows, signs


@jax.jit
def _subspace_pauli_map_nondiagonal(
    pauli_string: jax.Array,
    states: jax.Array
) -> tuple[jax.Array, jax.Array]:
    is_nondiagonal = jnp.packbits(jnp.not_equal(pauli_string % 3, 0).astype(np.uint8))
    is_signed = jnp.packbits(jnp.greater(pauli_string, 1).astype(np.uint8))
    mapped_states = states ^ is_nondiagonal
    rows = subspace_indices(mapped_states, states)
    signs = jnp.sum(jnp.bitwise_count(states & is_signed), axis=-1) % 2
    return rows, signs


@jax.jit
def subspace_indices(mapped_states: jax.Array, states: jax.Array) -> jax.Array:
    """Return the positions of the states in the subspace, or -1 if not found."""
    # Borrowing from jax._src.numpy.lax_numpy._searchsorted_via_sort
    # Attempted to replace the lax functions with corresponding jnp ones but that worsened the
    # memory leak by 2x
    def _rank(x):
        idx = lax.iota(np.int32, x.shape[0])
        # lax.sort seems to leak GPU memory; can lose as much as 5 GB when sorting x of shape (5M,9)
        sorted_idx = lax.sort(tuple(x.T) + (idx,), num_keys=x.shape[1])[-1]
        return array_creation.zeros_like(idx).at[sorted_idx].set(idx)

    index = _rank(lax.concatenate([mapped_states, states], 0))[:mapped_states.shape[0]]
    positions = lax.sub(index, _rank(mapped_states)).astype(np.int32)
    return jnp.where(jnp.all(jnp.equal(mapped_states, states[positions]), axis=1), positions, -1)


@jax.jit
def apply_pauli(vector, rows, elements):
    return (vector * jnp.expand_dims(elements, tuple(range(1, vector.ndim))))[rows]


@jax.jit
def apply_paulis(vector, rows, elements):
    return jnp.sum(jax.vmap(apply_pauli, in_axes=(None, 0, 0))(vector, rows, elements), axis=0)


@jax.jit
def apply_i(vector):
    return vector


@jax.jit
def apply_x(vector):
    return vector[::-1]


@jax.jit
def apply_y(vector):
    vector = vector[::-1] * 1.j
    return vector.at[0].multiply(-1.)


@jax.jit
def apply_z(vector):
    return vector.at[1].multiply(-1.)


@partial(jax.jit, static_argnames=['qubit'])
def apply_pauli_qubit(vector, pauli_index, *, qubit):
    vector = jnp.moveaxis(vector, qubit, 0)
    vector = jax.lax.switch(pauli_index, [apply_i, apply_x, apply_y, apply_z], vector)
    return jnp.moveaxis(vector, 0, qubit)


@jax.jit
def apply_pauli_string(vector, pauli_string):
    nq = pauli_string.shape[0]
    fns = [partial(apply_pauli_qubit, qubit=i) for i in range(nq)]
    vector = vector.reshape((2,) * nq)
    vector = jax.lax.fori_loop(0, nq,
                               lambda i, x: jax.lax.switch(i, fns, x, pauli_string[i]),
                               vector)
    return vector.reshape(-1)


apply_paulistring_mat = jax.jit(jax.vmap(apply_pauli_string, in_axes=(1, None), out_axes=1))
apply_paulistrings = jax.jit(jax.vmap(apply_pauli_string, in_axes=(None, 0)))
apply_paulistrings_mat = jax.jit(jax.vmap(apply_paulistrings, in_axes=(1, None), out_axes=1))


@jax.jit
def apply_paulistring_sum(vector, pauli_strings, coeffs):
    def accumulate(iloop, vec):
        return vec + coeffs[iloop] * apply_pauli_string(vector, pauli_strings[iloop])

    return jax.lax.fori_loop(0, pauli_strings.shape[0], accumulate, jnp.zeros_like(vector))


apply_paulistring_sum_mat = jax.jit(jax.vmap(apply_paulistring_sum,
                                             in_axes=(1, None, None), out_axes=1))
