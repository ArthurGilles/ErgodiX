from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax, random
import equinox as eqx

from ..distributions import TargetDistribution
from .params import AISParams
from .mala import _mala


def _single_ais(key: jax.Array,
                target: Callable,
                reference: TargetDistribution,
                dim: int,
                params: AISParams) -> Tuple:
    """
    Single AIS run producing one weighted sample.
    This function is vectorized over samples in the ``ais`` function.
    It is meant to be used inside the ``ais`` function and is not meant to be
    used directly.

    Parameters
    ----------
        key: jax.Array
            Random key for JAX random number generation.
        target: Callable
            Single-sample, scalar-valued (unnormalized) log-density of the
            target. A ``TargetDistribution`` satisfies this through its
            ``__call__``.
        reference: TargetDistribution
            Reference (proposal) distribution ``pi_0``. Must provide both a
            log-density (``__call__``) and a ``sample`` method, since AIS is
            initialised by an exact draw from it.
        dim: int
            Dimension of the distributions. Must match the reference.
        params: AISParams
            Parameters for the AIS algorithm.
    Returns
    -------
    x: jax.Array
        The final sample (approximately distributed as the target once
        reweighted by ``log_w``).
    log_w: jax.Array
        The scalar log importance weight of the sample.
    x_hist: jax.Array
        (only if ``params.return_history``) sample after each annealing step.
    accept_hist: jax.Array
        (only if ``params.return_history``) MALA acceptance rate at each step.
    """
    # value_and_grad of the two endpoint log-densities. Combined below to form
    # the intermediate (geometric-path) log-density and its score.
    log_pi_vg  = jax.value_and_grad(lambda x: target(x))
    log_ref_vg = jax.value_and_grad(lambda x: reference(x))

    betas  = params.schedule.betas(params.n_steps)   # (n_steps + 1,)
    beta_k = betas[1:]                                # (n_steps,) transition temperatures
    dbetas = jnp.diff(betas)                          # (n_steps,) log-weight increments

    key, key_x0, key_scan = random.split(key, num=3)

    # Initialise with an exact draw from the reference pi_0 = pi(beta = 0).
    x = reference.sample(key_x0, ())
    if x.shape[-1] != dim:
        raise ValueError(
            f"reference produces samples of dimension {x.shape[-1]}, but dim={dim}"
        )

    def step(carry: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
             inputs: Tuple[jax.Array, jax.Array]) -> Tuple:
        x, log_w, step_size, key = carry
        beta, dbeta = inputs
        key, key_mala = random.split(key)

        # weight at the incoming state x_{k-1}.
        # evaluated before the transition that targets pi_k.
        log_w = log_w + dbeta * (target(x) - reference(x))

        # MALA
        def log_and_score(y: jax.Array) -> Tuple[jax.Array, jax.Array]:
            lr, gr = log_ref_vg(y)
            lp, gp = log_pi_vg(y)
            return (1.0 - beta) * lr + beta * lp, (1.0 - beta) * gr + beta * gp

        x, accept_rate = _mala(key_mala, x, log_and_score, step_size, params)

        # Adapt the MALA step size
        if params.adapt_step_size:
            step_size = step_size * jnp.exp(
                params.learning_rate * (accept_rate - params.target_accept)
            )
            step_size = jnp.clip(step_size, min=params.step_min, max=params.step_max)

        return (x, log_w, step_size, key), (x, accept_rate)

    init = (x, jnp.zeros(()), params.step_size, key_scan)

    if params.return_history:
        (x, log_w, _, _), (x_hist, accept_hist) = lax.scan(step, init, (beta_k, dbetas))
        return x, log_w, x_hist, accept_hist
    else:
        def step_no_hist(carry, inputs):
            carry, _ = step(carry, inputs)
            return carry, None
        (x, log_w, _, _), _ = lax.scan(step_no_hist, init, (beta_k, dbetas))
        return x, log_w


# Vmap the single-item sampler over the batch of keys.
_ais = eqx.filter_vmap(
    _single_ais,
    in_axes=(0, None, None, None, None)
)


@eqx.filter_jit
def ais(key: jax.Array,
        target: Union[TargetDistribution, Callable[[jax.Array], jax.Array]],
        reference: TargetDistribution,
        batch_size: int,
        dim: int,
        params: AISParams) -> Tuple:
    """
    Annealed Importance Sampling (Neal, 2001) for a target distribution.

    AIS bridges an easy-to-sample reference ``pi_0`` and the target ``pi_n``
    with a ladder of geometric-path distributions
    ``pi_k proportional to pi_0 ** (1 - beta_k) * pi_n ** beta_k``. Each of the
    ``batch_size`` particles is drawn from the reference, carried through the
    ladder by MALA transitions, and accumulates an importance weight along the
    way. The returned ``(samples, log_weights)`` are *properly weighted* for the
    target: expectations under ``pi_n`` are estimated by the self-normalized
    average ``sum(w f(x)) / sum(w)`` with ``w = exp(log_weights)``, and the mean
    weight is an unbiased estimate of ``Z_n / Z_0`` (see
    ``log_normalizing_constant``).

    Parameters
    ----------
        key: jax.Array
            Random key for JAX random number generation.
        target: TargetDistribution or Callable[[jax.Array], jax.Array]
            The target ``pi_n`` to sample from, given by its (unnormalized)
            log-density. Either a ``TargetDistribution`` from
            ``ergodix.distributions`` (whose ``__call__`` is the log-density) or
            a plain callable mapping a single point of shape ``(dim,)`` to a
            scalar log-density. It is differentiated internally, so it must be
            single-sample and scalar-valued -- do not pre-``vmap`` it.
        reference: TargetDistribution
            Reference (proposal) distribution ``pi_0``. Must provide both a
            log-density (``__call__``) and a ``sample`` method, since AIS starts
            from an exact draw from it (e.g. an ``IsotropicGaussian``). For a
            correct ``Z_n / Z_0`` estimate the reference must be *normalized*;
            the weighted-sample distribution itself is unaffected by the
            reference normalisation.
        batch_size: int
            Number of independent weighted samples to generate.
        dim: int
            Dimension of the distributions. Must match the reference.
        params: AISParams
            Parameters for the AIS algorithm (see ``AISParams``).
    Returns
    -------
    samples: jax.Array
        The final samples, of shape ``(batch_size, dim)``.
    log_weights: jax.Array
        The log importance weights, of shape ``(batch_size,)``.
    x_hist: jax.Array
        (only if ``params.return_history``) samples after each annealing step,
        of shape ``(batch_size, n_steps, dim)``.
    accept_hist: jax.Array
        (only if ``params.return_history``) MALA acceptance rate at each step,
        of shape ``(batch_size, n_steps)``.

    Example
    -------
    >>> import jax, jax.numpy as jnp
    >>> from ergodix.distributions import IsotropicGaussian, IsotropicGMM
    >>> from ergodix.AIS import ais, AISParams, effective_sample_size
    >>> key = jax.random.PRNGKey(0)
    >>> target = IsotropicGMM(weights=jnp.ones(2),
    ...                       means=jnp.array([[-4., 0.], [4., 0.]]),
    ...                       variances=jnp.ones(2))
    >>> reference = IsotropicGaussian(mean=jnp.zeros(2), std=jnp.full((2,), 5.0))
    >>> params = AISParams(n_steps=64, n_mcmc_steps=5)
    >>> samples, log_w = ais(key, target, reference, batch_size=512, dim=2, params=params)
    >>> samples.shape, log_w.shape
    ((512, 2), (512,))
    """
    keys = random.split(key, batch_size)
    return _ais(keys, target, reference, dim, params)


def log_normalizing_constant(log_weights: jax.Array) -> jax.Array:
    """
    Estimate ``log(Z_n / Z_0)`` from a batch of AIS log-weights.

    The mean importance weight is an unbiased estimator of the ratio of
    normalizing constants ``Z_n / Z_0``; this returns its logarithm,
    ``logsumexp(log_weights) - log(N)``. It equals ``log Z_n`` when the
    reference ``pi_0`` is normalized (``Z_0 = 1``).

    Parameters
    ----------
    log_weights: jax.Array
        AIS log-weights of shape ``(batch_size,)`` as returned by ``ais``.
    Returns
    -------
    jax.Array
        Scalar estimate of ``log(Z_n / Z_0)``.
    """
    n = log_weights.shape[0]
    return jax.scipy.special.logsumexp(log_weights) - jnp.log(n)


def effective_sample_size(log_weights: jax.Array) -> jax.Array:
    """
    Effective sample size of a batch of AIS log-weights.

    ``ESS = (sum w) ** 2 / sum w ** 2`` lies in ``[1, batch_size]``; a value
    close to ``batch_size`` indicates well-balanced weights (a finer ladder or
    more MCMC steps raise it).

    Parameters
    ----------
    log_weights: jax.Array
        AIS log-weights of shape ``(batch_size,)`` as returned by ``ais``.
    Returns
    -------
    jax.Array
        Scalar effective sample size.
    """
    return jnp.exp(
        2.0 * jax.scipy.special.logsumexp(log_weights)
        - jax.scipy.special.logsumexp(2.0 * log_weights)
    )


def resample(key: jax.Array,
             samples: jax.Array,
             log_weights: jax.Array) -> jax.Array:
    """
    Multinomial resampling of weighted AIS samples into equally weighted ones.

    Draws ``batch_size`` indices with probabilities proportional to the
    importance weights, returning unweighted samples approximately distributed
    as the target.

    Parameters
    ----------
    key: jax.Array
        Random key for JAX random number generation.
    samples: jax.Array
        AIS samples of shape ``(batch_size, dim)``.
    log_weights: jax.Array
        AIS log-weights of shape ``(batch_size,)``.
    Returns
    -------
    jax.Array
        Resampled, equally weighted samples of shape ``(batch_size, dim)``.
    """
    n = log_weights.shape[0]
    idx = random.categorical(key, log_weights, shape=(n,))
    return samples[idx]
