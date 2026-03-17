"""Discrete wrapper and controlled observation corruption for Pointax."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .discrete_actions import discrete_to_continuous
from .utils import FAMILY_TO_ID, ScenarioSpec, stage_order


@dataclass
class WrapperState:
    base_state: object
    current_obs: np.ndarray
    previous_position: np.ndarray
    steps: int


class DiscretePointaxWrapper:
    """Python-side wrapper for Pointax discrete control and noisy observations."""

    def __init__(self, env, params, scenario: ScenarioSpec, action_scale: float = 0.6):
        self.env = env
        self.params = params
        self.scenario = scenario
        self.action_scale = action_scale
        self.state: Optional[WrapperState] = None
        self._obs_history: Deque[np.ndarray] = deque(maxlen=max(1, scenario.noise.observation_delay + 1))
        self._last_emitted_obs: Optional[np.ndarray] = None
        self._last_visible_goal: Optional[np.ndarray] = None
        self._obs_age = 0
        self._drift_bias = np.zeros(2, dtype=np.float32)

    @property
    def observation_dim(self) -> int:
        return 8

    def reset(self, key: jax.Array) -> Tuple[np.ndarray, Dict[str, float]]:
        obs, base_state = self.env.reset_env(key, self.params)
        obs_np = np.asarray(obs, dtype=np.float32)
        self._drift_bias = self._sample_drift_bias(key)
        self._obs_history.clear()
        for _ in range(self._obs_history.maxlen):
            self._obs_history.append(obs_np.copy())
        self._last_visible_goal = obs_np[4:6].copy()
        self._last_emitted_obs = None
        self._obs_age = 0
        wrapped = self._corrupt_observation(obs_np, key, force_refresh=True)
        self.state = WrapperState(
            base_state=base_state,
            current_obs=wrapped,
            previous_position=obs_np[:2].copy(),
            steps=0,
        )
        return wrapped, self._build_info(done=False, reward=0.0, success=False, wall_contact=False)

    def step(self, action_id: int, key: jax.Array) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        current_position = np.asarray(self.state.base_state.position, dtype=np.float32)
        current_velocity = np.asarray(self.state.base_state.velocity, dtype=np.float32)
        action = self._apply_action_noise(action_id, key)
        obs, base_state, reward, done, info = self.env.step_env(
            key,
            self.state.base_state,
            jnp.asarray(action, dtype=jnp.float32),
            self.params,
        )
        obs_np = np.asarray(obs, dtype=np.float32)
        new_position = obs_np[:2].copy()
        wall_contact = self._wall_contact_from_step(current_position, current_velocity, action, new_position)
        self._obs_history.append(obs_np.copy())
        wrapped_obs = self._corrupt_observation(obs_np, key, force_refresh=False)
        self.state = WrapperState(
            base_state=base_state,
            current_obs=wrapped_obs,
            previous_position=new_position,
            steps=self.state.steps + 1,
        )
        return wrapped_obs, float(reward), bool(done), self._build_info(
            done=bool(done),
            reward=float(reward),
            success=bool(info["is_success"]),
            wall_contact=wall_contact,
        )

    def current_true_position(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Wrapper is not initialized.")
        return np.asarray(self.state.base_state.position, dtype=np.float32)

    def current_goal(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Wrapper is not initialized.")
        return np.asarray(self.state.base_state.desired_goal, dtype=np.float32)

    def current_velocity(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Wrapper is not initialized.")
        return np.asarray(self.state.base_state.velocity, dtype=np.float32)

    def world_to_cell(self, position_xy: np.ndarray) -> Tuple[int, int]:
        cell = self.env._world_to_cell(jnp.asarray(position_xy, dtype=jnp.float32), self.params)
        return int(cell[0]), int(cell[1])

    def cell_to_world(self, cell: Tuple[int, int]) -> np.ndarray:
        row, col = cell
        world = self.env._cell_to_world(row, col, self.params)
        return np.asarray(world, dtype=np.float32)

    def free_cell_mask(self) -> np.ndarray:
        maze = np.asarray(self.params.maze_map)
        return maze != 1

    def _sample_drift_bias(self, key: jax.Array) -> np.ndarray:
        std = self.scenario.noise.drift_bias_std
        if std <= 0:
            return np.zeros(2, dtype=np.float32)
        drift = jax.random.normal(key, (2,)) * std
        return np.asarray(drift, dtype=np.float32)

    def _apply_action_noise(self, action_id: int, key: jax.Array) -> np.ndarray:
        action = discrete_to_continuous(action_id, scale=self.action_scale)
        std = self.scenario.noise.action_noise_std
        if std > 0:
            noise_key = jax.random.fold_in(key, 11)
            noise = np.asarray(jax.random.normal(noise_key, (2,)), dtype=np.float32) * std
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def _corrupt_observation(self, obs: np.ndarray, key: jax.Array, force_refresh: bool) -> np.ndarray:
        delayed = self._delayed_observation()
        obs_model = delayed.copy()
        noise_std = self.scenario.noise.obs_noise_std
        if noise_std > 0:
            noise_key = jax.random.fold_in(key, 17)
            obs_model[:4] += np.asarray(jax.random.normal(noise_key, (4,)), dtype=np.float32) * noise_std

        obs_model[:2] += self._drift_bias

        refresh_interval = max(1, self.scenario.noise.refresh_interval)
        should_refresh = force_refresh or self._last_emitted_obs is None or (self.state is None) or ((self.state.steps + 1) % refresh_interval == 0)
        if not should_refresh:
            self._obs_age += 1
            stale = self._last_emitted_obs.copy()
            stale[-1] = float(self._obs_age)
            return stale

        goal_visible = 1.0
        goal_key = jax.random.fold_in(key, 23)
        if self.scenario.noise.goal_mask_prob > 0:
            mask_draw = float(jax.random.uniform(goal_key))
            if mask_draw < self.scenario.noise.goal_mask_prob:
                goal_visible = 0.0
                obs_model[4:6] = 0.0
            else:
                self._last_visible_goal = obs_model[4:6].copy()
        else:
            self._last_visible_goal = obs_model[4:6].copy()

        if goal_visible == 0.0 and self._last_visible_goal is not None:
            obs_model[4:6] = 0.0

        self._obs_age = 0
        wrapped = np.concatenate([obs_model, np.array([goal_visible, 0.0], dtype=np.float32)], axis=0)
        self._last_emitted_obs = wrapped.copy()
        return wrapped

    def _delayed_observation(self) -> np.ndarray:
        delay = self.scenario.noise.observation_delay
        if delay <= 0 or len(self._obs_history) <= delay:
            return self._obs_history[-1].copy()
        return list(self._obs_history)[-delay - 1].copy()

    def _wall_contact_from_step(
        self,
        old_position: np.ndarray,
        old_velocity: np.ndarray,
        action: np.ndarray,
        new_position: np.ndarray,
    ) -> bool:
        velocity = np.clip(old_velocity, -float(self.params.max_velocity), float(self.params.max_velocity))
        acceleration = action * float(self.params.motor_gear) / float(self.params.mass)
        intended_velocity = np.clip(
            velocity + acceleration * float(self.params.dt),
            -float(self.params.max_velocity),
            float(self.params.max_velocity),
        )
        intended_position = old_position + intended_velocity * float(self.params.dt)
        correction_distance = float(np.linalg.norm(new_position - intended_position))
        if correction_distance > 1e-4:
            return True
        return self._touching_wall(new_position)

    def _touching_wall(self, position_xy: np.ndarray) -> bool:
        current_cell = self.env._world_to_cell(jnp.asarray(position_xy, dtype=jnp.float32), self.params)
        radius = float(self.params.robot_radius) + 1e-4
        half_size = float(self.params.maze_size_scaling) * 0.5
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                row = int(current_cell[0]) + di
                col = int(current_cell[1]) + dj
                if row < 0 or row >= int(self.params.map_length) or col < 0 or col >= int(self.params.map_width):
                    continue
                if int(self.params.maze_map[row, col]) != 1:
                    continue
                wall_center = np.asarray(self.env._cell_to_world(row, col, self.params), dtype=np.float32)
                aabb_min = wall_center - half_size
                aabb_max = wall_center + half_size
                closest = np.clip(position_xy, aabb_min, aabb_max)
                diff = position_xy - closest
                if float(np.linalg.norm(diff)) < radius:
                    return True
        return False

    def _build_info(self, done: bool, reward: float, success: bool, wall_contact: bool) -> Dict[str, float]:
        stage_id = stage_order().index(self.scenario.stage) if self.scenario.stage in stage_order() else -1
        return {
            "done": float(done),
            "reward": float(reward),
            "success": float(success),
            "wall_contact": float(wall_contact),
            "timeout": float(done and not success),
            "family_id": float(FAMILY_TO_ID[self.scenario.family]),
            "stage_id": float(stage_id),
            "stage_name": self.scenario.stage,
            "scenario_name": self.scenario.name,
        }
