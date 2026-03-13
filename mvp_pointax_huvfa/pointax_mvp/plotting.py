"""Plotting utilities for comparison plots and heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .envs import build_base_env
from .models import SharedQNetwork, q_values_for_all_options
from .utils import LEARNED_OPTIONS, ensure_dir


def plot_main_comparison(comparison_rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    frame = pd.DataFrame(comparison_rows)
    groups = list(frame["group"].unique())
    run_names = list(frame["run_name"].unique())
    x = np.arange(len(groups))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=140)
    for idx, run_name in enumerate(run_names):
        subset = frame[frame["run_name"] == run_name]
        subset = subset.set_index("group").reindex(groups).reset_index()
        axes[0].bar(x + (idx - 0.5) * width, subset["success_rate"], width=width, label=run_name)
        axes[1].bar(x + (idx - 0.5) * width, subset["normalized_efficiency"], width=width, label=run_name)
    axes[0].set_title("OOD Success Rate")
    axes[1].set_title("OOD Normalized Efficiency")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(groups, rotation=20)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_learning_curve(frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    for run_name, group in frame.groupby("run_name"):
        ax.plot(group["stage_index"], group["ood_success_rate"], marker="o", label=run_name)
    ax.set_xlabel("Stage / Budget Window")
    ax.set_ylabel("OOD Success Rate")
    ax.set_title("OOD Success Rate vs Training Progress")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_value_heatmaps(
    model: SharedQNetwork,
    params,
    scenario,
    action_scale: float,
    resolution: int,
    output_dir: Path,
) -> Dict[str, np.ndarray]:
    output_dir = ensure_dir(output_dir)
    bundle = build_base_env(scenario, max_steps=scenario.max_steps, action_scale=action_scale)
    wrapper = bundle.wrapper
    goal = wrapper.cell_to_world(scenario.goal_cell)
    maze = np.asarray(bundle.params.maze_map)
    x_min = -bundle.params.x_map_center
    x_max = bundle.params.x_map_center
    y_min = -bundle.params.y_map_center
    y_max = bundle.params.y_map_center
    xs = np.linspace(x_min, x_max, resolution, dtype=np.float32)
    ys = np.linspace(y_min, y_max, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    obs = np.stack(
        [
            xx.ravel(),
            yy.ravel(),
            np.zeros_like(xx).ravel(),
            np.zeros_like(xx).ravel(),
            np.full(xx.size, goal[0], dtype=np.float32),
            np.full(xx.size, goal[1], dtype=np.float32),
            np.ones(xx.size, dtype=np.float32),
            np.zeros(xx.size, dtype=np.float32),
        ],
        axis=-1,
    )
    goal_batch = np.repeat(goal[None, :], obs.shape[0], axis=0)
    q_values = q_values_for_all_options(
        model,
        params,
        jnp.asarray(obs, dtype=jnp.float32),
        jnp.asarray(goal_batch, dtype=jnp.float32),
    )
    option_maps: Dict[str, np.ndarray] = {}
    best_option = np.argmax(np.max(np.asarray(q_values), axis=-1), axis=-1).reshape(resolution, resolution)
    free_mask = np.zeros((resolution, resolution), dtype=bool)
    for row in range(resolution):
        for col in range(resolution):
            cell = wrapper.world_to_cell(np.array([xx[row, col], yy[row, col]], dtype=np.float32))
            free_mask[row, col] = maze[cell[0], cell[1]] != 1
    for option_id, option_name in enumerate(LEARNED_OPTIONS):
        option_value = np.max(np.asarray(q_values)[:, option_id, :], axis=-1).reshape(resolution, resolution)
        option_value = np.where(free_mask, option_value, np.nan)
        option_maps[option_name] = option_value
        fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
        mesh = ax.imshow(option_value, origin="lower", extent=[x_min, x_max, y_min, y_max], cmap="viridis")
        ax.set_title(f"{option_name} Value Heatmap")
        fig.colorbar(mesh, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"{option_name.lower()}_heatmap.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    pref = np.where(free_mask, best_option, np.nan)
    mesh = ax.imshow(pref, origin="lower", extent=[x_min, x_max, y_min, y_max], cmap="coolwarm")
    ax.set_title("Option Preference Map")
    fig.colorbar(mesh, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "option_preference.png")
    plt.close(fig)
    return option_maps
