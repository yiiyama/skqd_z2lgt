import jax
import jax.numpy as jnp
from rqutils.sqd import get_xsource


@jax.jit(static_argnames=['nplaq', 'max_order'])
def make_perturbation_series(nplaq, max_order, states_p, lengths):
    xsources = jax.lax.scan(
        lambda _, x: (None, get_xsource(x, states_p, sorted_input=False)),
        None,
        jnp.packbits(jnp.eye(nplaq, dtype=jnp.uint8), axis=1)
    )[1]
    ecoeffs = jnp.zeros(max_order // 2)
    acoeffs = jnp.zeros((max_order, states_p.shape[0]))
    single_plaq = jnp.sum(jnp.bitwise_count(states_p), axis=1) == 1
    div_2l = 0.5 / lengths
    acoeffs = acoeffs.at[0].set(single_plaq * div_2l)
    # [[single_plaq], [0], [0], ...]
    ecoeffs = ecoeffs.at[0].set(-acoeffs[0] @ single_plaq)
    # [E(2), 0, 0, ...]

    def add_ac(ix, val):
        ac, ac_prev = val
        ac += ac_prev.at[xsources[ix]].get(mode='fill', fill_value=0., wrap_negative_indices=False)
        return ac, ac_prev

    def compute(iloop, val):
        """Compute |Ω(2m)>, |Ω(2m+1)>, and E(2m+2) for m = iloop + 1."""
        ec, ac = val
        order = (iloop + 1) * 2
        aidx = order - 1
        vomega = jax.lax.fori_loop(0, nplaq, add_ac,
                                   (jnp.zeros(states_p.shape[0]), ac[aidx - 1]))[0]
        ac = ac.at[aidx].set((vomega + jnp.roll(ec[::-1], iloop) @ ac[1::2]) * div_2l)

        order += 1
        aidx = order - 1
        vomega = jax.lax.fori_loop(0, nplaq, add_ac,
                                   (jnp.zeros(states_p.shape[0]), ac[aidx - 1]))[0]
        ac = ac.at[aidx].set((vomega + jnp.roll(ec[::-1], iloop + 1) @ ac[::2]) * div_2l)

        ec = ec.at[iloop + 1].set(-ac[aidx] @ single_plaq)

        return ec, ac

    return jax.lax.fori_loop(0, max_order // 2 - 1, compute, (ecoeffs, acoeffs))
