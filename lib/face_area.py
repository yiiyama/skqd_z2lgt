"""Compute the area of contiguous plaquettes."""
from typing import Optional
import jax
import jax.numpy as jnp


@jax.jit
def compute_counts(amat: jax.Array, seeds: jax.Array, states: Optional[jax.Array] = None):
    """Compute the number of plaquettes connected to the initial list for given configurations.
    
    Args:
        amat: Adjacency matrix. Little endian ordering.
        initial: Initial n-hot vector representing the central plaquette(s). Little endian.
        states: List of state indices to compute the areas for. Big endian integers. If None, the
            full list of 2^N indices is used.

    Returns:
        Number of plaquettes connected to the initial list (included) for each state.
    """
    nbit = seeds.shape[0]
    ndim = 2 ** nbit
    if states is None:
        states = jnp.arange(ndim)

    # Convert the binary matrix into a 1D array of integers. The array is still little endian along
    # axis 0 but the integers are big-endian.
    amat = jnp.sum((1 << jnp.arange(nbit))[None, :] * amat, axis=1)
    # Similarly, convert the initial binary vector to an integer.
    seeds = jnp.sum((1 << jnp.arange(nbit)) * seeds)

    def get_counts(val):
        counts, neighbors, visited = val[:3]
        # Check coincident bits between the current list of neighbors and the states
        hits = states & neighbors
        # Add up the number of hits
        counts += jnp.bitwise_count(hits)
        # Binary array representing the next neighbors to consider for each state
        # Amounts to a left-multiplication of hits (for each state) by amat with OR as the summation
        neighbors_bin = (hits[:, None] & amat[None, :] != 0).astype(neighbors.dtype)
        # Convert the binary arrays (little endian because amat axis 0 is) to integers
        neighbors = jnp.sum(neighbors_bin << jnp.arange(nbit), axis=1)
        # Mask out the visited plaquettes
        neighbors &= ~visited
        # Continue if visited flag is not saturated
        not_done = visited != ndim - 1
        # Update the list of visited plaquettes
        visited |= jnp.sum((visited & amat != 0).astype(neighbors.dtype) << jnp.arange(nbit))
        return counts, neighbors, visited, not_done

    return jax.lax.while_loop(
        lambda val: val[3],
        get_counts,
        (
            jnp.zeros(ndim, dtype=int),
            jnp.full(ndim, seeds),
            seeds,
            True
        )
    )[0]
