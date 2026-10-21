"""Training evaluation metrics."""
from collections.abc import Callable
import jax.numpy as jnp


def success_rate(num_gen) -> Callable:
    def fn(model, u_batch, v_batch, _):
        samples = model.sample(u_batch, num_gen)  # size comes at axis 0
        test = jnp.all(jnp.equal(v_batch[None, ...], samples), axis=-1)
        return jnp.mean(jnp.any(test, axis=0).astype(int))

    return fn


def l2_weights(model, u_batch, v_batch, updates):
    return (jnp.mean(jnp.square(model.weights_vu))
            + jnp.mean(jnp.square(model.weights_hu))
            + jnp.mean(jnp.square(model.weights_hv)))


def l2_biases(model, u_batch, v_batch, updates):
    return jnp.mean(jnp.square(model.bias_v)) + jnp.mean(jnp.square(model.bias_h))
