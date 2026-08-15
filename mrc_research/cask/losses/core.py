"""Loss primitives with explicit stop-gradient and censoring semantics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cask.masking import masked_mean


def diagonal_gaussian_kl(
    posterior_mean,
    posterior_logvar,
    prior_mean,
    prior_logvar,
    *,
    detach_posterior: bool,
):
    if detach_posterior:
        posterior_mean = jax.lax.stop_gradient(posterior_mean)
        posterior_logvar = jax.lax.stop_gradient(posterior_logvar)
    variance_ratio = jnp.exp(posterior_logvar - prior_logvar)
    mean_term = jnp.square(prior_mean - posterior_mean) * jnp.exp(-prior_logvar)
    per_dimension = 0.5 * (
        prior_logvar - posterior_logvar + variance_ratio + mean_term - 1
    )
    return jnp.sum(per_dimension, axis=-1)


def standard_normal_kl(mean, logvar):
    return 0.5 * jnp.sum(jnp.exp(logvar) + jnp.square(mean) - 1 - logvar, axis=-1)


def fsq_coordinate_cross_entropy(logits, integer_codes, levels: tuple[int, ...]):
    if logits.shape[1] != len(levels):
        raise ValueError("The coordinate axis must match the FSQ levels.")
    if len(set(levels)) != 1 or logits.shape[-1] != levels[0]:
        raise ValueError("The initial CE implementation expects equal FSQ levels.")
    half = (levels[0] - 1) // 2
    target_digits = jax.lax.stop_gradient(integer_codes.astype(jnp.int32) + half)
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    selected = jnp.take_along_axis(
        log_probabilities,
        target_digits[..., None],
        axis=-1,
    )[..., 0]
    return -jnp.mean(selected, axis=-1)


def discrete_hazard_nll(hazard_logits, event_target, at_risk_mask):
    """Bernoulli hazard NLL.

    event_target is true only for an observed normal termination. Censored
    timeout/failure examples contain no positive target; at_risk_mask stays true
    through their final observed instant.
    """

    if hazard_logits.shape != event_target.shape or hazard_logits.shape != at_risk_mask.shape:
        raise ValueError("hazard_logits, event_target, and at_risk_mask must match.")
    target = event_target.astype(hazard_logits.dtype)
    per_step = (
        jax.nn.softplus(hazard_logits)
        - target * hazard_logits
    )
    return masked_mean(per_step, at_risk_mask, axis=-1)
