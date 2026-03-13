"""Discrete action space on top of Pointax continuous actions."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


ACTION_NAMES: Tuple[str, ...] = (
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "UP_LEFT",
    "UP_RIGHT",
    "DOWN_LEFT",
    "DOWN_RIGHT",
    "HOLD",
)

ACTION_VECTORS: Dict[int, np.ndarray] = {
    0: np.array([0.0, 1.0], dtype=np.float32),
    1: np.array([0.0, -1.0], dtype=np.float32),
    2: np.array([-1.0, 0.0], dtype=np.float32),
    3: np.array([1.0, 0.0], dtype=np.float32),
    4: np.array([-1.0, 1.0], dtype=np.float32),
    5: np.array([1.0, 1.0], dtype=np.float32),
    6: np.array([-1.0, -1.0], dtype=np.float32),
    7: np.array([1.0, -1.0], dtype=np.float32),
    8: np.array([0.0, 0.0], dtype=np.float32),
}


def num_actions() -> int:
    return len(ACTION_NAMES)


def action_name(action_id: int) -> str:
    return ACTION_NAMES[action_id]


def discrete_to_continuous(action_id: int, scale: float = 0.6) -> np.ndarray:
    vector = ACTION_VECTORS[int(action_id)].astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-6:
        vector = vector / norm
    return vector * np.float32(scale)


def direction_to_action(delta_xy: np.ndarray) -> int:
    dx, dy = float(delta_xy[0]), float(delta_xy[1])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 8

    if abs(dx) > 1e-6 and abs(dy) > 1e-6:
        if dx > 0 and dy > 0:
            return 5
        if dx < 0 and dy > 0:
            return 4
        if dx > 0 and dy < 0:
            return 7
        return 6

    if abs(dx) >= abs(dy):
        return 3 if dx > 0 else 2
    return 0 if dy > 0 else 1
