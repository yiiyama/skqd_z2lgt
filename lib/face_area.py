"""Compute the area of contiguous plaquettes."""
from typing import Optional
import jax
import jax.numpy as jnp


@jax.jit
def compute_areas(amat: jax.Array, seeds: jax.Array, states: Optional[jax.Array] = None):
    """Compute the number of plaquettes connected to the initial list for given configurations.
    
    Args:
        amat: Adjacency matrix. Little endian ordering.
        seeds: Initial n-hot vector representing the seed plaquette(s). Little endian.
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

    def grow_clusters(val):
        clusters, visited = val
        clusters |= jnp.sum(
            (clusters[:, None] & amat[None, :] != 0).astype(clusters.dtype) << jnp.arange(nbit),
            axis=1
        )
        clusters &= states
        visited |= jnp.sum((visited & amat != 0).astype(clusters.dtype) << jnp.arange(nbit))
        return clusters, visited

    clusters = jax.lax.while_loop(
        lambda val: val[1] != ndim - 1,
        grow_clusters,
        (states & seeds, seeds)
    )[0]

    return jnp.bitwise_count(clusters)
