"""Evaluation helpers for greedy hierarchical inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .envs import build_base_env
from .models import SharedQNetwork, q_values_for_all_options
from .teacher import shortest_path
from .utils import LEARNED_OPTIONS, ScenarioSpec


@dataclass
class EpisodeResult:
    success: float
    steps: int
    timeout: float
    wall_contacts: float
    efficiency: float
    option_counts: Dict[str, int]


def greedy_option_and_action(model: SharedQNetwork, params, obs: np.ndarray, goal_xy: np.ndarray) -> Tuple[int, int]:
    obs_batch = jnp.asarray(obs[None, :], dtype=jnp.float32)
    goal_batch = jnp.asarray(goal_xy[None, :], dtype=jnp.float32)
    q_values = q_values_for_all_options(model, params, obs_batch, goal_batch)[0]
    option_scores = jnp.max(q_values, axis=-1)
    option_id = int(jnp.argmax(option_scores))
    action_id = int(jnp.argmax(q_values[option_id]))
    return option_id, action_id


def run_greedy_episode(
    scenario: ScenarioSpec,
    model: SharedQNetwork,
    params,
    seed: int,
    action_scale: float,
) -> EpisodeResult:
    bundle = build_base_env(scenario, max_steps=scenario.max_steps, action_scale=action_scale)
    wrapper = bundle.wrapper
    key = jax.random.PRNGKey(seed)
    obs, _ = wrapper.reset(key)
    option_counts = {name: 0 for name in LEARNED_OPTIONS}
    done = False
    wall_contacts = 0.0
    steps = 0
    while not done and steps < scenario.max_steps:
        option_id, action_id = greedy_option_and_action(model, params, obs, wrapper.current_goal())
        option_counts[LEARNED_OPTIONS[option_id]] += 1
        key = jax.random.fold_in(key, steps + 1)
        obs, _, done, info = wrapper.step(action_id, key)
        wall_contacts += float(info["wall_contact"])
        steps += 1

    free_mask = wrapper.free_cell_mask()
    shortest = len(shortest_path(free_mask, scenario.start_cell, scenario.goal_cell)) - 1
    efficiency = float(shortest / max(steps, 1))
    success = float(info["success"]) if steps > 0 else 0.0
    timeout = float(done and not bool(success))
    return EpisodeResult(
        success=success,
        steps=steps,
        timeout=timeout,
        wall_contacts=wall_contacts,
        efficiency=efficiency,
        option_counts=option_counts,
    )


def summarize_group(results: Sequence[EpisodeResult]) -> Dict[str, float]:
    success = np.array([item.success for item in results], dtype=np.float32)
    steps = np.array([item.steps for item in results], dtype=np.float32)
    timeout = np.array([item.timeout for item in results], dtype=np.float32)
    wall_contacts = np.array([item.wall_contacts for item in results], dtype=np.float32)
    efficiency = np.array([item.efficiency for item in results], dtype=np.float32)
    summary = {
        "success_rate": float(success.mean()) if len(success) else 0.0,
        "steps_to_goal": float(steps[success > 0].mean()) if np.any(success > 0) else float(steps.mean() if len(steps) else 0.0),
        "timeout_rate": float(timeout.mean()) if len(timeout) else 0.0,
        "wall_contacts": float(wall_contacts.mean()) if len(wall_contacts) else 0.0,
        "normalized_efficiency": float(efficiency.mean()) if len(efficiency) else 0.0,
    }
    for option_name in LEARNED_OPTIONS:
        summary[f"option_usage_{option_name.lower()}"] = float(
            np.mean([result.option_counts[option_name] for result in results]) if results else 0.0
        )
    return summary


def evaluate_scenarios(
    model: SharedQNetwork,
    params,
    eval_sets: Mapping[str, Sequence[ScenarioSpec]],
    seed: int,
    action_scale: float,
    episodes_per_scenario: int,
) -> Dict[str, Dict[str, float]]:
    output: Dict[str, Dict[str, float]] = {}
    seed_cursor = seed
    for group_name, scenarios in eval_sets.items():
        group_results: List[EpisodeResult] = []
        for scenario in scenarios:
            for _ in range(episodes_per_scenario):
                group_results.append(
                    run_greedy_episode(
                        scenario=scenario,
                        model=model,
                        params=params,
                        seed=seed_cursor,
                        action_scale=action_scale,
                    )
                )
                seed_cursor += 1
        output[group_name] = summarize_group(group_results)
    return output
