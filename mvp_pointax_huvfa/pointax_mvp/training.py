"""Offline teacher-driven training entrypoints for the Pointax MVP."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import flax
from flax.serialization import to_bytes
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .evaluation import evaluate_scenarios
from .losses import critic_loss
from .models import SharedQNetwork
from .plotting import make_value_heatmaps
from .replay import ReplayBuffer
from .task_library import build_eval_scenarios, build_training_scenarios
from .teacher import collect_teacher_trajectory
from .utils import (
    LEARNED_OPTIONS,
    TrainConfig,
    ensure_dir,
    flatten_metrics,
    numpy_rng,
    save_csv,
    save_json,
    stage_order,
    timestamp,
)


@dataclass
class CriticTrainBundle:
    train_state: TrainState
    target_params: flax.core.FrozenDict
    model: SharedQNetwork


def stage_trajectory_count(config: TrainConfig, stage_index: int) -> int:
    multiplier = config.stage_fractions[stage_index] * len(stage_order())
    return max(1, int(round(config.trajectories_per_scenario * multiplier)))


def stage_update_budget(config: TrainConfig) -> List[int]:
    total = config.updates_per_stage * len(stage_order())
    raw = [int(round(total * frac)) for frac in config.stage_fractions]
    delta = total - sum(raw)
    raw[-1] += delta
    return [max(1, value) for value in raw]


def build_model(config: TrainConfig) -> SharedQNetwork:
    return SharedQNetwork(
        hidden_dims=config.hidden_dims,
        embedding_dim=config.embedding_dim,
        num_options=len(LEARNED_OPTIONS),
        num_actions=9,
    )


def init_train_bundle(config: TrainConfig) -> CriticTrainBundle:
    model = build_model(config)
    rng = jax.random.PRNGKey(config.seed)
    sample_obs = jnp.zeros((1, 8), dtype=jnp.float32)
    sample_goal = jnp.zeros((1, 2), dtype=jnp.float32)
    sample_option = jnp.zeros((1,), dtype=jnp.int32)
    sample_action = jnp.zeros((1,), dtype=jnp.int32)
    params = model.init(rng, sample_obs, sample_goal, sample_option, sample_action)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(config.learning_rate))
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return CriticTrainBundle(train_state=train_state, target_params=params, model=model)


def _train_step(bundle: CriticTrainBundle, batch, config: TrainConfig) -> Tuple[CriticTrainBundle, Mapping[str, float]]:
    def loss_fn(params):
        return critic_loss(
            params=params,
            target_params=bundle.target_params,
            model=bundle.model,
            batch=batch,
            gamma=config.gamma,
            teacher_weight=config.teacher_weight,
            use_hcrl_aux=config.use_hcrl_aux,
            hcrl_aux_weight=config.hcrl_aux_weight,
            contrastive_temperature=config.contrastive_temperature,
            contrastive_logsumexp_penalty=config.contrastive_logsumexp_penalty,
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(bundle.train_state.params)
    train_state = bundle.train_state.apply_gradients(grads=grads)
    new_target = optax.incremental_update(train_state.params, bundle.target_params, step_size=1.0 / max(config.target_update_period, 1))
    updated = CriticTrainBundle(train_state=train_state, target_params=new_target, model=bundle.model)
    flat_metrics = {key: float(value) for key, value in metrics.items()}
    flat_metrics["loss"] = float(loss)
    return updated, flat_metrics


def collect_stage_data(
    scenarios: Sequence[object],
    trajectories_per_scenario: int,
    config: TrainConfig,
    seed_offset: int,
) -> ReplayBuffer:
    buffer = ReplayBuffer()
    seed_cursor = seed_offset
    for scenario in scenarios:
        collected = 0
        attempts = 0
        max_attempts = max(trajectories_per_scenario * config.teacher_retry_limit, trajectories_per_scenario)
        while collected < trajectories_per_scenario and attempts < max_attempts:
            trajectory = collect_teacher_trajectory(
                scenario=scenario,
                seed=seed_cursor,
                action_scale=config.action_scale,
                max_steps=config.max_steps_per_episode,
            )
            if trajectory:
                buffer.add_trajectory(scenario, trajectory)
                collected += 1
            seed_cursor += 1
            attempts += 1
    return buffer


def merge_replay(target: ReplayBuffer, source: ReplayBuffer) -> None:
    offset = len(target.rows)
    target.rows.extend(source.rows)
    target.trajectory_ranges.extend((start + offset, stop + offset) for start, stop in source.trajectory_ranges)


def save_checkpoint(run_dir: Path, train_bundle: CriticTrainBundle, stage_name: str) -> None:
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    with (checkpoint_dir / f"{stage_name}.msgpack").open("wb") as handle:
        handle.write(to_bytes(train_bundle.train_state.params))
    with (checkpoint_dir / "latest.msgpack").open("wb") as handle:
        handle.write(to_bytes(train_bundle.train_state.params))


def update_latest_alias(run_dir: Path) -> None:
    latest_dir = run_dir.parent / "latest"
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    try:
        os.symlink(run_dir.name, latest_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(run_dir, latest_dir)


def train(config: TrainConfig) -> Path:
    training_scenarios = build_training_scenarios()
    eval_sets = build_eval_scenarios()
    run_dir = ensure_dir(Path(config.results_root) / config.run_name / timestamp())
    save_json(run_dir / "config.json", config.to_dict())
    train_bundle = init_train_bundle(config)
    global_buffer = ReplayBuffer()
    rng = numpy_rng(config.seed)
    learning_rows: List[Mapping[str, float]] = []
    eval_rows: List[Mapping[str, float]] = []
    stage_metrics_rows: List[Mapping[str, float]] = []
    update_budgets = stage_update_budget(config)

    if config.training_mode == "no_curriculum":
        mixed_buffer = ReplayBuffer()
        for stage_index, stage_name in enumerate(stage_order()):
            stage_buffer = collect_stage_data(
                scenarios=training_scenarios[stage_name],
                trajectories_per_scenario=stage_trajectory_count(config, stage_index),
                config=config,
                seed_offset=config.seed + 100 * (stage_index + 1),
            )
            merge_replay(mixed_buffer, stage_buffer)
        mixed_buffer.relabel_future_observations(rng)
        global_buffer = mixed_buffer

    for stage_index, stage_name in enumerate(stage_order()):
        if config.training_mode == "manual_curriculum":
            stage_buffer = collect_stage_data(
                scenarios=training_scenarios[stage_name],
                trajectories_per_scenario=stage_trajectory_count(config, stage_index),
                config=config,
                seed_offset=config.seed + 1000 * (stage_index + 1),
            )
            stage_buffer.relabel_future_observations(rng)
            merge_replay(global_buffer, stage_buffer)

        if len(global_buffer) == 0:
            raise RuntimeError("Replay buffer is empty; teacher data collection failed.")

        for update_idx in range(update_budgets[stage_index]):
            batch = global_buffer.sample(rng, min(config.batch_size, len(global_buffer)))
            train_bundle, metrics = _train_step(train_bundle, batch, config)
            metrics_row = {"stage_index": stage_index, "stage_name": stage_name, "update_idx": update_idx}
            metrics_row.update(metrics)
            stage_metrics_rows.append(metrics_row)

        if (stage_index + 1) % config.eval_every == 0:
            summary = evaluate_scenarios(
                model=train_bundle.model,
                params=train_bundle.train_state.params,
                eval_sets=eval_sets,
                seed=config.seed + 5000 + stage_index,
                action_scale=config.action_scale,
                episodes_per_scenario=config.num_eval_episodes,
            )
            ood_success = float(np.mean([group["success_rate"] for group in summary.values()]))
            learning_rows.append({"run_name": config.run_name, "stage_index": stage_index + 1, "ood_success_rate": ood_success})
            for group_name, group_metrics in summary.items():
                eval_row = {"run_name": config.run_name, "stage_index": stage_index + 1, "group": group_name}
                eval_row.update(group_metrics)
                eval_rows.append(eval_row)

        if (stage_index + 1) % config.checkpoint_every == 0:
            save_checkpoint(run_dir, train_bundle, stage_name)

    final_eval = evaluate_scenarios(
        model=train_bundle.model,
        params=train_bundle.train_state.params,
        eval_sets=eval_sets,
        seed=config.seed + 9999,
        action_scale=config.action_scale,
        episodes_per_scenario=config.num_eval_episodes,
    )
    aggregate_metrics = {
        "run_name": config.run_name,
        "training_mode": config.training_mode,
        "replay_size": len(global_buffer),
    }
    aggregate_metrics.update(flatten_metrics("", final_eval))
    save_json(run_dir / "metrics.json", aggregate_metrics)
    save_csv(run_dir / "metrics.csv", stage_metrics_rows)
    save_csv(run_dir / "learning_curve.csv", learning_rows)
    save_csv(run_dir / "eval_summary.csv", eval_rows)
    save_csv(run_dir / "replay_summary.csv", global_buffer.as_rows())

    composed_scenario = build_eval_scenarios()["composed_ood"][0]
    make_value_heatmaps(
        model=train_bundle.model,
        params=train_bundle.train_state.params,
        scenario=composed_scenario,
        action_scale=config.action_scale,
        resolution=config.heatmap_resolution,
        output_dir=run_dir / "plots",
    )
    update_latest_alias(run_dir)
    return run_dir
