"""Helpers for building wrapped Pointax environments from scenario specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import pointax

from .utils import ScenarioSpec
from .wrappers import DiscretePointaxWrapper


@dataclass
class ScenarioEnvBundle:
    env: pointax.PointMazeEnv
    params: pointax.EnvParams
    wrapper: DiscretePointaxWrapper


def build_base_env(scenario: ScenarioSpec, max_steps: int, action_scale: float) -> ScenarioEnvBundle:
    env = pointax.make_custom(
        [list(row) for row in scenario.maze_layout],
        maze_id=scenario.name,
        reward_type="sparse",
    )
    params = env.default_params.replace(
        position_noise_range=0.0,
        max_steps_in_episode=max_steps,
    )
    wrapper = DiscretePointaxWrapper(
        env=env,
        params=params,
        scenario=scenario,
        action_scale=action_scale,
    )
    return ScenarioEnvBundle(env=env, params=params, wrapper=wrapper)


def seed_to_key(seed: int) -> jax.Array:
    return jax.random.PRNGKey(seed)
