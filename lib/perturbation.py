import jax
import jax.numpy as jnp
from rqutils.sqd import get_xsource


@jax.jit(static_argnames=['nplaq', 'max_order'])
def make_perturbation_series(nplaq, max_order, states, lengths):
    nc2 = (nplaq * (nplaq - 1)) // 2
    pq_indices = jnp.nonzero(states[nplaq:nplaq + nc2].T, size=nc2 * 2)[1].reshape(nplaq, nplaq - 1)
    pq_indices += nplaq

    states_p = jnp.packbits(states, axis=1)
    xsources = jax.lax.scan(
        lambda _, x: (None, get_xsource(x, states_p, sorted_input=False)),
        None,
        jnp.packbits(jnp.eye(nplaq, dtype=jnp.uint8), axis=1)
    )[1]
    energies = jnp.zeros(max_order // 2)
    coeffs = jnp.zeros((max_order, states.shape[0]))
    single_plaq = jnp.sum(states, axis=1) == 1
    coeffs = coeffs.at[0].set(single_plaq * 0.5 / lengths)

    def compute_pfactor(order, energies, coeffs):
        pfactor = jnp.sum(coeffs[order - 3, pq_indices], axis=1)
        pfactor += energies @ jnp.roll(coeffs[1::2, :nplaq][::-1], order // 2, axis=0)
        return pfactor

    def compute_energy(order, energies, coeffs):
        """Compute E^{(o)}."""
        pfactor = jax.lax.cond(
            jnp.equal(order, 2),
            lambda o, e, c: jnp.ones(nplaq),
            compute_pfactor,
            order, energies, coeffs
        )    
        return energies.at[order // 2 - 1].set(-coeffs[0, :nplaq] @ pfactor)

    def compute_order(order, energies, coeffs):
        energies = jax.lax.cond(
            jnp.equal(order & 1, 0),
            compute_energy,
            lambda o, e, c: e,
            order, energies, coeffs
        )

        coeff = jnp.sum(
            coeffs[order - 2].at[xsources].get(mode='fill', fill_value=0.,
                                               wrap_negative_indices=False),
            axis=0
        )
        coeff += energies @ jnp.roll(coeffs[::2][::-1], (order - 1) // 2, axis=0)
        coeff *= 0.5 / lengths
        coeffs = coeffs.at[order - 1].set(coeff)
        return energies, coeffs

    return jax.lax.fori_loop(
        2, max_order + 1,
        lambda i, v: compute_order(i, *v),
        (energies, coeffs)
    )
