"""Compute the area of contiguous plaquettes."""
from typing import Optional
import jax
import jax.numpy as jnp


@jax.jit
def make_clusters(amat: jax.Array, seeds: jax.Array, states: Optional[jax.Array] = None):
    """Identify clusters of plaquettes including the given seeds in each plaquette configuration.
    
    Args:
        amat: Adjacency matrix. Little endian ordering.
        seeds: Initial n-hot vector representing the seed plaquette(s). Little endian.
        states: List of state indices (plaquette configurations) to find the clusters from. Big
            endian integers. If None, the full list of 2^N indices is used.

    Returns:
        Clusters of plaquettes encoded as integers.
    """
    nbit = amat.shape[0]
    if states is None:
        states = jnp.arange(2 ** nbit)

    if amat.ndim == 2:
        # If amat is given as a binary matrix, convert to a list of integers.
        amat = _convert_le_array_to_be_int(amat)
    if seeds.ndim == 1 and seeds.dtype == jnp.uint8:
        # Similarly, convert the initial binary vector to an integer.
        seeds = _convert_le_array_to_be_int(seeds)

    def grow_clusters(val):
        clusters, visited = val
        clusters |= jnp.sum(
            (clusters[:, None] & amat != 0).astype(clusters.dtype) << jnp.arange(nbit),
            axis=1
        )
        clusters &= states
        # visited flags can be a scalar or a vector (one flag per state)
        visited |= jnp.sum(
            (visited[..., None] & amat != 0).astype(clusters.dtype) << jnp.arange(nbit),
            axis=-1
        )
        return clusters, visited

    return jax.lax.while_loop(
        lambda val: jnp.any(jnp.logical_and(val[1] != 0, val[1] != 2 ** nbit - 1)),
        grow_clusters,
        (states & seeds, seeds)
    )[0]


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
    clusters = make_clusters(amat, seeds, states=states)
    return jnp.bitwise_count(clusters)


@jax.jit
def count_windings(
    amat: jax.Array,
    peripheries: jax.Array,
    seed: int,
    states: Optional[jax.Array] = None
):
    """Compute the winding numbers around a seed.
    
    Args:
        amat: Adjacency matrix. Little endian ordering.
        peripheries: N-hot vector representing plaquettes at the periphery of the lattice. Little
            endian.
        seed: Plaquette index of the seed.
        states: List of state indices to compute the windings for. Big endian integers. If None, the
            full list of 2^N indices is used.

    Returns:
        Winding numbers of the states.
    """
    nbit = amat.shape[0]
    if amat.ndim == 2:
        amat = _convert_le_array_to_be_int(amat)
    if peripheries.ndim == 1:
        peripheries = _convert_le_array_to_be_int(peripheries)
    if states is None:
        states = jnp.arange(2 ** nbit)

    def add_winds(val):
        windings, seeds, _states = val
        clusters = make_clusters(amat, seeds, states=_states)
        windings += (clusters != 0).astype(int)
        clusters *= ((clusters & peripheries) != peripheries).astype(int)
        neighbors = jnp.sum(
            (clusters[:, None] & amat[None, :] != 0).astype(clusters.dtype) << jnp.arange(nbit),
            axis=1
        )
        neighbors &= ~clusters
        surroundings = make_clusters(amat, neighbors, states=~_states)
        seeds = (clusters | surroundings) * (jnp.bitwise_count(surroundings & peripheries) == 0)
        _states |= seeds
        return windings, seeds, _states

    return jax.lax.while_loop(
        lambda val: jnp.any(val[1]),
        add_winds,
        (jnp.zeros_like(states), jnp.full_like(states, 1 << seed), states)
    )[0]


def _convert_le_array_to_be_int(arr: jax.Array, npmod=jnp):
    """Convert the binary matrix into a 1D array of integers.
    
    The array is still little endian along axis 0 but the integers are big-endian.
    """
    return npmod.sum(arr * (1 << npmod.arange(arr.shape[-1])), axis=-1)
