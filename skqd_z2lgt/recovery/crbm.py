"""Conditional restricted Boltzmann machine."""
import logging
from typing import Optional
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from flax import nnx

LOG = logging.getLogger(__name__)


class ConditionalRBM(nnx.Module):
    """Conditional restricted Boltzmann machine."""
    def __init__(self, num_u: int, num_v: int, num_h: int, *, rngs: Optional[nnx.Rngs] = None):
        if rngs is None:
            rngs = nnx.Rngs(params=0, sample=0)

        init = nnx.initializers.normal(1.)
        self.weights_vu = nnx.Param(init(rngs.params(), (num_v, num_u), jnp.float64))
        self.weights_hu = nnx.Param(init(rngs.params(), (num_h, num_u), jnp.float64))
        self.weights_hv = nnx.Param(init(rngs.params(), (num_h, num_v), jnp.float64))
        self.bias_v = nnx.Param(init(rngs.params(), (num_v,), jnp.float64))
        self.bias_h = nnx.Param(init(rngs.params(), (num_h,), jnp.float64))
        self.rngs = rngs
        self.therm_steps = 100
        self.vhat_size = 100

    def copy(self) -> 'ConditionalRBM':
        model = ConditionalRBM(
            self.weights_vu.shape[1],
            self.bias_v.shape[0],
            self.bias_h.shape[0],
            rngs=nnx.Rngs(params=self.rngs.params, sample=self.rngs.sample)
        )
        params = nnx.split(self, nnx.Param, ...)[1]
        for key, param in params.items():
            getattr(model, key).value = param.value

        model.therm_steps = self.therm_steps
        model.vhat_size = self.vhat_size

        return model

    def save(self, file: str | h5py.Group):
        params_state, rngs_state = map(nnx.pure, nnx.state(self, nnx.Param, ...))
        if isinstance(file, str):
            out = h5py.File(file, 'w', libver='latest')
        else:
            out = file

        out.create_dataset('therm_steps', data=self.therm_steps)
        out.create_dataset('vhat_size', data=self.vhat_size)

        params = out.create_group('params')
        for key, value in params_state.items():
            params.create_dataset(key, data=value)

        rngs = out.create_group('rngs')
        for key, state in rngs_state['rngs'].items():
            group = rngs.create_group(key)
            group.create_dataset('count', data=state['count'])
            group.create_dataset('key', data=jax.random.key_data(state['key']))

        if isinstance(file, str):
            out.close()

    @staticmethod
    def load(file: str | h5py.Group) -> 'ConditionalRBM':
        if isinstance(file, str):
            source = h5py.File(file, libver='latest')
        else:
            source = file

        params = {key: jnp.array(data[()]) for key, data in source['params'].items()}
        rngs_state = {}
        for key, state in source['rngs'].items():
            rngs_state[key] = {}
            try:
                rngs_state[key]['count'] = jnp.array(state['count'][()])
            except KeyError:
                LOG.error('Failed to load rngs/%s/count', key)
                rngs_state[key]['count'] = jnp.array(0, dtype=np.uint32)
            try:
                rngs_state[key]['key'] = jax.random.wrap_key_data(state['key'][()])
            except KeyError:
                LOG.error('Failed to load rngs/%s/key', key)
                rngs_state[key]['key'] = jnp.array([0, 0], dtype=np.uint32)
        therm_steps = source['therm_steps'][()]
        vhat_size = source['vhat_size'][()]

        if isinstance(file, str):
            source.close()

        rngs = nnx.Rngs(**{key: state['key'] for key, state in rngs_state.items()})
        for key, state in rngs_state.items():
            getattr(rngs, key).count.value = state['count']

        model = ConditionalRBM(
            params['weights_vu'].shape[1],
            params['bias_v'].shape[0],
            params['bias_h'].shape[0],
            rngs=rngs
        )
        for key, value in params.items():
            getattr(model, key).value = value

        model.therm_steps = therm_steps
        model.vhat_size = vhat_size

        return model

    @nnx.jit
    def energy(self, u_state: jax.Array, v_state: jax.Array, h_state: jax.Array) -> jax.Array:
        """Energy function."""
        e_val = -(h_state[:, None, :] @ self.weights_hv @ v_state[:, :, None]
                  + v_state[:, None, :] @ self.weights_vu @ u_state[:, :, None]
                  + h_state[:, None, :] @ self.weights_hu @ u_state[:, :, None])
        e_val = jnp.squeeze(e_val, axis=(1, 2))
        e_val -= v_state @ self.bias_v + h_state @ self.bias_h
        return e_val

    @nnx.jit
    def free_energy(self, u_state: jax.Array, v_state: jax.Array) -> jax.Array:
        """Free energy -log(sum_h(exp(-E(v,h,u))))."""
        f_val = -jnp.vecdot(v_state,
                            jnp.tensordot(u_state, self.weights_vu, (-1, 1)) + self.bias_v)
        f_val -= jnp.sum(
            jnp.log(1. + jnp.exp(jnp.tensordot(v_state, self.weights_hv, (-1, 1))
                                 + jnp.tensordot(u_state, self.weights_hu, (-1, 1))
                                 + self.bias_h)),
            axis=-1
        )
        return f_val

    @nnx.jit
    def conditional_logz(self, u_state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Log partition function given a u state. Intractable for num_v >~ 25.

        Since the raw sum of exp(-F) can be numerically unmanageable, we subtract the mean of F per
        sample before taking the exponential. This normalization factor will then be needed to
        compute the conditional probability and is therefore given as the second return value of
        this function.
        """
        num_v = self.bias_v.shape[0]
        all_v = ((jnp.arange(2 ** num_v)[:, None] >> jnp.arange(num_v)[None, :])
                 % 2).astype(np.uint8)
        all_f = self.free_energy(u_state, all_v)
        norm = jnp.mean(all_f, axis=0)
        all_f -= norm[None, ...]
        return jnp.log(jnp.sum(jnp.exp(-all_f), axis=0)), norm

    @nnx.jit
    def conditional_nll(self, u_state: jax.Array, v_state: jax.Array) -> jax.Array:
        """Probability of the v state given u state. Intractable for num_v >~ 25."""
        logz, norm = self.conditional_logz(u_state)
        return self.free_energy(u_state, v_state) - norm + logz

    @nnx.jit
    def h_activation(self, u_state: jax.Array, v_state: jax.Array) -> jax.Array:
        delta_e = jnp.tensordot(v_state, self.weights_hv, (-1, 1))
        delta_e += jnp.tensordot(u_state, self.weights_hu, (-1, 1))
        delta_e += self.bias_h
        return nnx.sigmoid(delta_e)

    @nnx.jit
    def v_activation(self, u_state: jax.Array, h_state: jax.Array) -> jax.Array:
        delta_e = jnp.tensordot(u_state, self.weights_vu, (-1, 1))
        delta_e += jnp.tensordot(h_state, self.weights_hv, (-1, 0))
        delta_e += self.bias_v
        return nnx.sigmoid(delta_e)

    @nnx.jit(static_argnames=['size', 'final_state_only', 'impl', 'unroll'])
    def gibbs_sample(
        self,
        u_state: jax.Array,
        init_v_state: jax.Array,
        size: Optional[int | tuple[int, ...]] = None,
        final_state_only: bool = False,
        impl: str = 'tensordot',
        unroll: int | bool = 1
    ) -> jax.Array:
        """MCMC sample generation."""
        if size is None:
            size = 1
            final_state_only = True
        if not isinstance(size, tuple):
            size = (int(size),)

        def loop_body(v_state, cmp):
            if impl == 'tensordot':
                h_state = jnp.tensordot(v_state, self.weights_hv, (-1, 1)) > cmp[0]
                v_state = jnp.tensordot(h_state, self.weights_hv, (-1, 0)) > cmp[1]
            else:
                # mul+sum somehow executes much faster (tensordot has a huge overhead in initial
                # kernel launch..??) but compiles slower when the u_state batch size is large (??)
                # P.S. Actually compilation is slow on A100 but not on V100
                h_state = jnp.sum(v_state[..., None, :] * self.weights_hv, axis=-1) > cmp[0]
                v_state = jnp.sum(h_state[..., None] * self.weights_hv, axis=-2) > cmp[1]
            if final_state_only:
                out = None
            else:
                out = v_state.astype(np.uint8)
            return v_state.astype(self.bias_v.dtype), out

        uwb_h = jnp.tensordot(u_state, self.weights_hu, (-1, 1)) + self.bias_h
        uwb_v = jnp.tensordot(u_state, self.weights_vu, (-1, 1)) + self.bias_v
        batch_dims = (np.prod(size).astype(int),) + u_state.shape[:-1]
        cmps = (
            self._activation_comparator(batch_dims + self.bias_h.shape) - uwb_h,
            self._activation_comparator(batch_dims + self.bias_v.shape) - uwb_v
        )

        final_v, v_states = jax.lax.scan(loop_body, init_v_state.astype(self.bias_v.dtype), xs=cmps,
                                         unroll=unroll)
        if final_state_only:
            return final_v.astype(np.uint8)
        return v_states

    def _activation_comparator(self, shape):
        """Comparator to determine the activation of v and h vectors.

        Activation functions are of form f=1/(1 + e^{-x}) where x is the linear transform of v / h.
        Instead of comparing a sample u from the uniform distribution to f, we precompute
        -log(1/u-1) and compare it to x.
        """
        uniform = jax.random.uniform(self.rngs.sample(), shape, minval=1.e-12)
        return -jnp.log(1. / uniform - 1.)

    @nnx.jit
    def percloss_states(self, u_state: jax.Array):
        v_states = self.sample(u_state, self.vhat_size)
        free_energies = self.free_energy(u_state, v_states)
        min_indices = jnp.argmin(free_energies, axis=0)
        return v_states[min_indices, jnp.arange(u_state.shape[0])]

    @nnx.jit
    def percloss(
        self,
        u_state: jax.Array,
        v_state: jax.Array,
        vhat_state: jax.Array
    ) -> jax.Array:
        return self.free_energy(u_state, v_state) - self.free_energy(u_state, vhat_state)

    @nnx.jit
    def meanloss(
        self,
        u_state: jax.Array,
        v_state: jax.Array,
        vg_states: jax.Array
    ) -> jax.Array:
        return (self.free_energy(u_state, v_state)
                - jnp.mean(self.free_energy(u_state, vg_states), axis=0))

    @nnx.jit(static_argnames=['size'])
    def sample(
        self,
        u_state: jax.Array,
        size: Optional[int | tuple[int, ...]] = None
    ) -> jax.Array:
        """Generate samples of v vectors.

        Returns the result of Gibbs sampling from a thermalized state. Thermalization is performed
        by `self.therm_steps` iterations of Gibbs sampling on the initial v vector drawn from
        v_activation(u, 0).

        Args:
            u_state: Condition bits. Shape [..., num_u].
            size: Number of samples to generate for each condition vector.

        Returns:
            Array of v vector samples with shape [*size, *u_state.shape[:-1], num_v].
        """
        pv = nnx.sigmoid(jnp.tensordot(u_state, self.weights_vu, (-1, 1)) + self.bias_v)
        uniform = jax.random.uniform(self.rngs.sample(), pv.shape)
        v_state = (uniform < pv).astype(np.uint8)
        v_state = self.gibbs_sample(u_state, v_state, size=self.therm_steps, final_state_only=True)
        return self.gibbs_sample(u_state, v_state, size=size)
