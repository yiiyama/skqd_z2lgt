"""Gaussian kernel density estimation for sharded one-dimensional data."""
from functools import partial
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=['num_point', 'batch_size'])
def gaussian_kde(data: jax.Array, num_point: int, batch_size: int = 16):
    """Gaussian kernel density estimator of a 1D (possibly sharded) array.

    The estimator is evaluated at num_point points evenly spaced between min(data) and max(data),
    with the bandwidth determined by Scott's rule (same as scipy.stats.gaussian_kde):
    h = std(data, ddof=1) * n^{-1/5}.

    All reductions over the data axis are plain jnp operations, so with a sharded input XLA
    computes per-shard partial results and combines them with all-reduce; the data are never
    gathered onto one device. The evaluation points are processed in batches of batch_size to
    bound the intermediate memory at batch_size * (local shard size).

    Args:
        data: 1D array of shape (n,), with any sharding.
        num_point: Number of evaluation points.
        batch_size: Number of evaluation points processed at once.

    Returns:
        Tuple (points, density) of arrays of shape (num_point,). The density is replicated.
    """
    n = data.shape[0]
    data = data.astype(jnp.result_type(data.dtype, jnp.float32))
    points = jnp.linspace(jnp.min(data), jnp.max(data), num_point)

    # Scott's rule
    mean = jnp.mean(data)
    std = jnp.sqrt(jnp.sum(jnp.square(data - mean)) / (n - 1))
    bandwidth = std * n ** (-1. / 5.)

    def eval_point(x):
        z = (x - data) / bandwidth
        return jnp.sum(jnp.exp(-0.5 * jnp.square(z)))

    kernel_sum = jax.lax.map(eval_point, points, batch_size=min(batch_size, num_point))
    density = kernel_sum / (n * bandwidth * jnp.sqrt(2. * jnp.pi))
    return points, density
