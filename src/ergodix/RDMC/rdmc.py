from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax, random

import equinox as eqx

from ..distributions import TargetDistribution
from .params import RDMCParams


def _single_RDMC(key: jax.Array,
                 target: TargetDistribution,
                 dim: int,
                 params: RDMCParams) -> Tuple:
    
    def score_single(y: jax.Array, x: jax.Array, t: float) -> jax.Array:
        c = jnp.exp(-(params.T - t))
        return target.score(y) + c * (x - c * y) / (1 - c**2)

    log_pdf = jax.vmap(target)
    score = jax.vmap(score_single, in_axes=(0, None, None))

    key, key_x, key_main = random.split(key,num=3)

    X = jnp.sqrt(1-jnp.exp(-2*params.T)) * random.normal(key_x,dim)

    def step(carry: Tuple[jax.Array, jax.Array],
             i: int) -> Tuple:
        
        key, x_curr = carry

        key, key_p, key_sample, key_ula, key_step = random.split(key,num=5)
        
        # Precompute useful constants
        dt = params.T/params.n_steps
        t = i*dt
        c = jnp.exp(-(params.T-t))
        e = jnp.exp(dt)
        
        # Use Importance Sampling to draw MC samples
        # BEGIN IS
        mean = x_curr/c
        var = (1/c**2-1)
        noise = random.normal(key_p, (params.n_mc_samples*params.n_particles, dim))
        particles = mean+jnp.sqrt(var)*noise #(n_mc_samples*n_particles,)
        
        #log_weights = log_q(particles, x_curr, t) - jnp.sum(noise**2,axis=-1)/2 #(n_mc_samples*n_particles,)

        log_weights = log_pdf(particles) #(n_mc_samples*n_particles,)
        log_weights = log_weights.reshape(params.n_mc_samples, params.n_particles) #(n_mc_samples, n_particles)
        particles_reshaped = particles.reshape(params.n_mc_samples, params.n_particles, dim) #(n_mc_samples, n_particles, dim)

        indices = jax.random.categorical(key_sample, log_weights, axis=1) # (n_mc_samples,)
        batch_idx = jnp.arange(params.n_mc_samples) # (n_mc_samples,)

        mc_samples = particles_reshaped[batch_idx, indices] # Shape: (n_mc_samples, dim)
        # END IS

        # Samples now drawn with IS
        # Use the ULA to correct the samples
        def ula_step(i: int, 
                     val: Tuple[jax.Array, jax.Array]) -> Tuple[jax.Array, jax.Array]:
            key, sample_curr = val
            key, key_n = random.split(key, num=2)
            new = sample_curr + \
                  params.ula_step_size*score(sample_curr, x_curr, t) + \
                  jnp.sqrt(2*params.ula_step_size)*random.normal(key_n, shape=sample_curr.shape)
            return (key, new)
        
        _, mc_samples = lax.fori_loop(0, params.n_ula_steps, ula_step, (key_ula, mc_samples))

        # Compute the score and integrate the SDE
        c = jnp.exp(-(params.T - t))
        score_est = jnp.mean(2*(c*mc_samples - x_curr)/(1 - c**2), axis=0)


        new_x = e*x_curr+(e-1)*score_est+jnp.sqrt(e**2-1)*random.normal(key_step, dim)

        return (key, new_x), (new_x, mc_samples)

    if params.return_history:
        (_, X), (X_hist, samples_hist ) = lax.scan(
            step, (key_main, X), jnp.arange(params.n_steps)
        )

        return X, X_hist, samples_hist
    else:
        def fori_main_step(i: int,
                           carry: Tuple[jax.Array, jax.Array]) -> Tuple[jax.Array, jax.Array]:
            carry, _ = step(carry, i)
            return carry
        
        (_, X) = lax.fori_loop(
            0, params.n_steps, fori_main_step, (key_main, X)
        )
        return X


# Vmap over the batch of keys
_RDMC = eqx.filter_vmap(
    _single_RDMC, 
    in_axes=(0, None, None, None)
)


@eqx.filter_jit
def RDMC(key: jax.Array,
         target: TargetDistribution,
         batch_size: int,
         dim: int,
         params: RDMCParams) -> Tuple:

    keys = random.split(key, batch_size)
    return _RDMC(keys, target, dim, params)