"""Flax models for the Pointax H-UVFA MVP."""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    dims: Sequence[int]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for dim in self.dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        return x


class SharedQNetwork(nn.Module):
    """Shared option-conditioned UVF critic."""

    hidden_dims: Sequence[int]
    embedding_dim: int
    num_options: int
    num_actions: int

    def setup(self) -> None:
        self.state_encoder = MLP((self.embedding_dim, self.embedding_dim))
        self.future_encoder = MLP((self.embedding_dim, self.embedding_dim))
        self.goal_encoder = MLP((self.embedding_dim, self.embedding_dim))
        self.option_embed = nn.Embed(self.num_options, self.embedding_dim)
        self.action_embed = nn.Embed(self.num_actions, self.embedding_dim)
        self.fusion = MLP(self.hidden_dims)
        self.q_head = nn.Dense(1)
        self.future_projection = nn.Dense(self.embedding_dim)
        self.goal_projection = nn.Dense(self.embedding_dim)

    def encode_state(self, obs: jax.Array) -> jax.Array:
        return self.state_encoder(obs)

    def encode_future_obs(self, future_obs: jax.Array) -> jax.Array:
        return self.future_encoder(future_obs)

    def encode_goal(self, goal_xy: jax.Array) -> jax.Array:
        return self.goal_encoder(goal_xy)

    def interaction_embedding(
        self,
        obs: jax.Array,
        goal_xy: jax.Array,
        option_id: jax.Array,
        action_id: jax.Array,
    ) -> jax.Array:
        state_embedding = self.encode_state(obs)
        goal_embedding = self.encode_goal(goal_xy)
        option_embedding = self.option_embed(option_id)
        action_embedding = self.action_embed(action_id)
        fused = jnp.concatenate(
            [state_embedding, goal_embedding, option_embedding, action_embedding],
            axis=-1,
        )
        return self.fusion(fused)

    def __call__(
        self,
        obs: jax.Array,
        goal_xy: jax.Array,
        option_id: jax.Array,
        action_id: jax.Array,
    ) -> dict:
        joint = self.interaction_embedding(obs, goal_xy, option_id, action_id)
        goal_embedding = self.encode_goal(goal_xy)
        return {
            "q": jnp.squeeze(self.q_head(joint), axis=-1),
            "soa": self.future_projection(joint),
            "goal": self.goal_projection(goal_embedding),
        }


def q_values_for_option(
    model: SharedQNetwork,
    params,
    obs: jax.Array,
    goal_xy: jax.Array,
    option_id: jax.Array,
) -> jax.Array:
    action_ids = jnp.arange(model.num_actions, dtype=jnp.int32)

    def eval_action(action_id: jax.Array) -> jax.Array:
        batch_actions = jnp.full((obs.shape[0],), action_id, dtype=jnp.int32)
        output = model.apply(params, obs, goal_xy, option_id, batch_actions)
        return output["q"]

    return jax.vmap(eval_action)(action_ids).transpose(1, 0)


def q_values_for_all_options(
    model: SharedQNetwork,
    params,
    obs: jax.Array,
    goal_xy: jax.Array,
) -> jax.Array:
    option_ids = jnp.arange(model.num_options, dtype=jnp.int32)

    def eval_option(option_id: jax.Array) -> jax.Array:
        batch_options = jnp.full((obs.shape[0],), option_id, dtype=jnp.int32)
        return q_values_for_option(model, params, obs, goal_xy, batch_options)

    return jax.vmap(eval_option)(option_ids).transpose(1, 0, 2)
