from collections.abc import Callable
import jax
import jax.numpy as jnp
from flax import nnx
from skqd_z2lgt.recovery.crbm import ConditionalRBM

LossFn = Callable[[ConditionalRBM, jax.Array, jax.Array], jax.Array]


@nnx.jit
def cd_percloss(
    model: ConditionalRBM,
    u_state: jax.Array,
    v_state: jax.Array
):
    vhat_state = jax.lax.stop_gradient(model.percloss_states(u_state))
    return jnp.mean(model.percloss(u_state, v_state, vhat_state))


@nnx.jit(static_argnames=['impl', 'unroll'])
def cd_meanloss(
    model: ConditionalRBM,
    u_state: jax.Array,
    v_state: jax.Array,
    impl: str = 'tensordot',
    unroll: int = 1
):
    assert model.vhat_size % unroll == 0
    vg_states = jax.lax.stop_gradient(
        model.gibbs_sample(u_state, v_state, model.vhat_size, impl=impl, unroll=unroll)
    )
    return jnp.mean(model.meanloss(u_state, v_state, vg_states))


@nnx.jit
def nll_loss(
    model: ConditionalRBM,
    u_state: jax.Array,
    v_state: jax.Array
):
    return jnp.mean(model.conditional_nll(u_state, v_state))


def make_l2_loss_fn(loss_fn: LossFn, l2w_weights: float, l2w_biases: float) -> LossFn:
    """Construct a loss function with L2 regularization."""
    def fn(
        model: ConditionalRBM,
        u_state: jax.Array,
        v_state: jax.Array
    ):
        loss = loss_fn(model, u_state, v_state)
        loss += l2w_weights * (
            jnp.mean(jnp.square(model.weights_vu))
            + jnp.mean(jnp.square(model.weights_hu))
            + jnp.mean(jnp.square(model.weights_hv))
        )
        loss += l2w_biases * (
            jnp.mean(jnp.square(model.bias_v))
            + jnp.mean(jnp.square(model.bias_h))
        )
        return loss

    return nnx.jit(fn)
