"""Shared data structures and utilities for the Pointax MVP."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import yaml


LEARNED_OPTIONS: Tuple[str, ...] = ("SEGMENT_FOLLOW", "JUNCTION_RESOLVE")
TEACHER_CONTROLLERS: Tuple[str, ...] = ("RECOVER_RECENTER", "A_STAR_NAVIGATION")
OPTION_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(LEARNED_OPTIONS)}
FAMILY_TO_ID: Dict[str, int] = {
    "A1": 0,
    "A2": 1,
    "A3": 2,
    "B1": 3,
    "B2": 4,
    "E1": 5,
    "E2": 6,
    "E3": 7,
}


@dataclass(frozen=True)
class NoiseSpec:
    obs_noise_std: float = 0.0
    goal_mask_prob: float = 0.0
    action_noise_std: float = 0.0
    observation_delay: int = 0
    refresh_interval: int = 1
    drift_bias_std: float = 0.0


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    family: str
    stage: str
    split: str
    maze_layout: Tuple[Tuple[Any, ...], ...]
    start_cell: Tuple[int, int]
    goal_cell: Tuple[int, int]
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    max_steps: int = 150
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["maze_layout"] = [list(row) for row in self.maze_layout]
        return payload


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 7
    gamma: float = 0.97
    learning_rate: float = 3e-4
    batch_size: int = 128
    hidden_dims: Tuple[int, ...] = (128, 128)
    embedding_dim: int = 32
    num_epochs: int = 10
    updates_per_stage: int = 200
    target_update_period: int = 20
    teacher_weight: float = 0.05
    hcrl_aux_weight: float = 0.05
    use_hcrl_aux: bool = False
    action_scale: float = 0.6
    max_steps_per_episode: int = 150
    trajectories_per_scenario: int = 6
    stage_fractions: Tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
    checkpoint_every: int = 1
    eval_every: int = 1
    num_eval_episodes: int = 4
    heatmap_resolution: int = 80
    run_name: str = "default"
    training_mode: str = "manual_curriculum"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TrainConfig":
        hidden_dims = tuple(payload.get("hidden_dims", cls.hidden_dims))
        stage_fractions = tuple(payload.get("stage_fractions", cls.stage_fractions))
        merged = dict(payload)
        merged["hidden_dims"] = hidden_dims
        merged["stage_fractions"] = stage_fractions
        return cls(**merged)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def mvp_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: os.PathLike) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def load_yaml_config(path: os.PathLike) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    extends = raw.pop("extends", None)
    if extends:
        parent = load_yaml_config(config_path.parent / extends)
        parent.update(raw)
        return parent
    return raw


def save_json(path: os.PathLike, payload: Mapping[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_csv(path: os.PathLike, rows: Sequence[Mapping[str, Any]]) -> None:
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def flatten_metrics(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            output.update(flatten_metrics(full_key, value))
        else:
            output[full_key] = float(value)
    return output


def stage_order() -> Tuple[str, ...]:
    return ("A1", "A2", "A3", "B1", "B2")


def chunked(sequence: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(sequence), chunk_size):
        yield sequence[start : start + chunk_size]


def numpy_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def as_numpy(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)
