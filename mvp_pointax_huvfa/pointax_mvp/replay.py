"""Replay structures for offline training and relabeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .utils import FAMILY_TO_ID, ScenarioSpec, stage_order


@dataclass
class TransitionBatch:
    obs: np.ndarray
    next_obs: np.ndarray
    goal_xy: np.ndarray
    option_id: np.ndarray
    action_id: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    stage_id: np.ndarray
    family_id: np.ndarray
    success: np.ndarray
    wall_contact: np.ndarray
    future_obs: Optional[np.ndarray] = None


class ReplayBuffer:
    def __init__(self) -> None:
        self.rows: List[Dict[str, np.ndarray]] = []
        self.trajectory_ranges: List[tuple[int, int]] = []

    def add_trajectory(self, scenario: ScenarioSpec, transitions: Sequence[object]) -> None:
        stage_id = stage_order().index(scenario.stage) if scenario.stage in stage_order() else -1
        family_id = FAMILY_TO_ID[scenario.family]
        start_index = len(self.rows)
        for transition in transitions:
            self.rows.append(
                {
                    "obs": np.asarray(transition.obs, dtype=np.float32),
                    "next_obs": np.asarray(transition.next_obs, dtype=np.float32),
                    "goal_xy": np.asarray(transition.goal_xy, dtype=np.float32),
                    "option_id": np.asarray(int(transition.option_id), dtype=np.int32),
                    "action_id": np.asarray(int(transition.action_id), dtype=np.int32),
                    "reward": np.asarray(float(transition.reward), dtype=np.float32),
                    "done": np.asarray(float(transition.done), dtype=np.float32),
                    "stage_id": np.asarray(stage_id, dtype=np.int32),
                    "family_id": np.asarray(family_id, dtype=np.int32),
                    "success": np.asarray(float(transition.success), dtype=np.float32),
                    "wall_contact": np.asarray(float(transition.wall_contact), dtype=np.float32),
                }
            )
        self.trajectory_ranges.append((start_index, len(self.rows)))

    def relabel_future_observations(self, rng: np.random.Generator) -> None:
        for start, stop in self.trajectory_ranges:
            for row_idx in range(start, stop):
                future_index = int(rng.integers(row_idx, stop))
                self.rows[row_idx]["future_obs"] = self.rows[future_index]["next_obs"]

    def __len__(self) -> int:
        return len(self.rows)

    def sample(self, rng: np.random.Generator, batch_size: int) -> TransitionBatch:
        indices = rng.integers(0, len(self.rows), size=batch_size)
        return self.batch(indices)

    def batch(self, indices: Sequence[int]) -> TransitionBatch:
        columns: Dict[str, List[np.ndarray]] = {}
        for index in indices:
            row = self.rows[int(index)]
            for key, value in row.items():
                columns.setdefault(key, []).append(value)
        stacked = {key: np.stack(value, axis=0) for key, value in columns.items()}
        return TransitionBatch(**stacked)

    def as_rows(self) -> List[Mapping[str, float]]:
        output: List[Mapping[str, float]] = []
        for row in self.rows:
            output.append(
                {
                    "reward": float(row["reward"]),
                    "done": float(row["done"]),
                    "stage_id": int(row["stage_id"]),
                    "family_id": int(row["family_id"]),
                    "option_id": int(row["option_id"]),
                    "action_id": int(row["action_id"]),
                    "success": float(row["success"]),
                    "wall_contact": float(row["wall_contact"]),
                }
            )
        return output
