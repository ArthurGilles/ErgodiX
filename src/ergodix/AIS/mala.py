from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import random, lax

from .params import AISParams


def _mala(key: jax.Array,
          init_x: jax.Array,
          log_and_score: Callable,
          step_size: jax.Array,
          params: AISParams) -> Tuple[jax.Array, jax.Array]:
    """
    Run a single MALA chain leaving the distribution defined by
    ``log_and_score`` invariant, and return its final state together with the
    empirical acceptance rate.

    This is the MCMC transition used at each intermediate distribution of the
    AIS algorithm. It is meant to be used inside the AIS algorithm and is not
    meant to be used directly.

    Parameters
    ----------
        key: jax.Array
            Random key for JAX random number generation.
        init_x: jax.Array
            Initial position of the chain, of shape ``(dim,)``.
        log_and_score: Callable
            Function mapping a position to the tuple ``(log_pdf, score)`` of the
            (unnormalized) invariant distribution, where ``score`` is the
            gradient of ``log_pdf``.
        step_size: jax.Array
            Step size of the MALA proposal.
        params: AISParams
            Parameters of the AIS algorithm (only ``n_mcmc_steps`` is used).
    Returns
    -------
    final_x: jax.Array
        Final position of the chain after ``n_mcmc_steps`` steps.
    accept_rate: jax.Array
        Empirical acceptance rate over the chain, a scalar in ``[0, 1]``.
    """

    def step(carry: Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
             _: None) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], None]:
        sample, score, log_pdf, accepts, key = carry
        key, key_dW, key_u = random.split(key, num=3)

        # Langevin proposal.
        dW = random.normal(key_dW, sample.shape)
        sample_new = sample + step_size * score + jnp.sqrt(2 * step_size) * dW

        # Log-density and score at the proposal, then the MH log-ratio.
        log_pdf_new, score_new = log_and_score(sample_new)
        d = sample - sample_new - step_size * score_new

        log_q     = -1.0 / (4 * step_size) * jnp.sum(d ** 2)
        log_q_new = -0.5 * jnp.sum(dW ** 2)

        log_alpha = log_pdf_new - log_pdf + log_q - log_q_new

        # Accept or reject.
        accept = jnp.log(random.uniform(key_u)) < log_alpha

        sample  = jnp.where(accept, sample_new, sample)
        # Carry the score and log-density forward so accepted proposals are
        # never re-evaluated.
        score   = jnp.where(accept, score_new, score)
        log_pdf = jnp.where(accept, log_pdf_new, log_pdf)
        accepts += accept.astype(accepts.dtype)

        return (sample, score, log_pdf, accepts, key), None

    init_log_pdf, init_score = log_and_score(init_x)
    (final_x, _, _, accepts, _), _ = lax.scan(
        step, (init_x, init_score, init_log_pdf, 0.0, key), length=params.n_mcmc_steps
    )

    return final_x, accepts / params.n_mcmc_steps
