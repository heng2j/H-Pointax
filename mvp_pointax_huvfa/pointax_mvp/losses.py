"""Loss functions for offline H-UVFA training."""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
import optax

from .models import SharedQNetwork, q_values_for_all_options, q_values_for_option


def info_nce(anchor: jax.Array, positive: jax.Array, temperature: float = 0.1) -> tuple[jax.Array, jax.Array]:
    anchor = anchor / (jnp.linalg.norm(anchor, axis=-1, keepdims=True) + 1e-8)
    positive = positive / (jnp.linalg.norm(positive, axis=-1, keepdims=True) + 1e-8)
    logits = jnp.matmul(anchor, positive.T) / temperature
    labels = jnp.arange(anchor.shape[0], dtype=jnp.int32)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(loss), logits


def critic_loss(
    params,
    target_params,
    model: SharedQNetwork,
    batch,
    gamma: float,
    teacher_weight: float,
    use_hcrl_aux: bool,
    hcrl_aux_weight: float,
    contrastive_temperature: float,
    contrastive_logsumexp_penalty: float,
) -> tuple[jax.Array, Dict[str, jax.Array]]:
    outputs = model.apply(params, batch.obs, batch.goal_xy, batch.option_id, batch.action_id)
    q_pred = outputs["q"]
    next_q_values = q_values_for_all_options(model, target_params, batch.next_obs, batch.goal_xy)
    next_best = jnp.max(next_q_values, axis=(1, 2))
    td_target = batch.reward + gamma * (1.0 - batch.done) * next_best
    td_loss = jnp.mean(jnp.square(q_pred - jax.lax.stop_gradient(td_target)))

    teacher_logits = q_values_for_option(model, params, batch.obs, batch.goal_xy, batch.option_id)
    teacher_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(teacher_logits, batch.action_id)
    )

    aux_loss = jnp.array(0.0, dtype=jnp.float32)
    aux_future_loss = jnp.array(0.0, dtype=jnp.float32)
    aux_goal_loss = jnp.array(0.0, dtype=jnp.float32)
    if use_hcrl_aux and getattr(batch, "future_obs", None) is not None:
        future_embedding = model.apply(target_params, batch.future_obs, method=model.encode_future_obs)
        goal_embedding = model.apply(target_params, batch.goal_xy, method=model.encode_goal)
        aux_future_loss, future_logits = info_nce(
            outputs["future_query"],
            jax.lax.stop_gradient(future_embedding),
            temperature=contrastive_temperature,
        )
        aux_goal_loss, goal_logits = info_nce(
            outputs["goal_query"],
            jax.lax.stop_gradient(goal_embedding),
            temperature=contrastive_temperature,
        )
        future_logsumexp = jax.nn.logsumexp(future_logits + 1e-6, axis=1)
        goal_logsumexp = jax.nn.logsumexp(goal_logits + 1e-6, axis=1)
        aux_loss = (
            aux_future_loss
            + aux_goal_loss
            + contrastive_logsumexp_penalty * (jnp.mean(future_logsumexp**2) + jnp.mean(goal_logsumexp**2))
        )

    total_loss = td_loss + teacher_weight * teacher_loss + hcrl_aux_weight * aux_loss
    metrics = {
        "loss": total_loss,
        "td_loss": td_loss,
        "teacher_loss": teacher_loss,
        "aux_loss": aux_loss,
        "aux_future_loss": aux_future_loss,
        "aux_goal_loss": aux_goal_loss,
        "q_mean": jnp.mean(q_pred),
        "target_mean": jnp.mean(td_target),
    }
    return total_loss, metrics
